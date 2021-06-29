import os
import time
import argparse
import math
import numpy as np
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import load_model
from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
from utils import to_gpu
import time
from math import e

from tqdm import tqdm
import layers
from utils import load_wav_to_torch
from scipy.io.wavfile import read

import os.path

from metric import alignment_metric

save_file_check_path = "save"
num_workers_ = 1 # DO NOT CHANGE WHEN USING TRUNCATION
start_from_checkpoints_from_zero = 0
gen_new_mels = 0

def create_mels():
    stft = layers.TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length,
                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                hparams.mel_fmax)

    def save_mel(file):
        audio, sampling_rate = load_wav_to_torch(file)
        if sampling_rate != stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(file, 
                sampling_rate, stft.sampling_rate))
        audio_norm = audio / hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0).cpu().numpy()
        np.save(file.replace('.wav', ''), melspec)

    import glob
    wavs = glob.glob('/media/cookie/Samsung 860 QVO/ClipperDatasetV2/**/*.wav',recursive=True)
    print(str(len(wavs))+" files being converted to mels")
    for index, i in tqdm(enumerate(wavs), smoothing=0, total=len(wavs)):
        if index < 0: continue
        try: save_mel(i)
        except Exception as ex: tqdm.write(i, " failed to process\n",ex,"\n")
    assert 0


class StreamingMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams, saved_lookup):
    # Get data, data loaders and collate function ready
    speaker_ids = saved_lookup if hparams.use_saved_speakers else None
    trainset = TextMelLoader(hparams.training_files, hparams, check_files=hparams.check_files, shuffle=False,
                           speaker_ids=speaker_ids)
    valset = TextMelLoader(hparams.validation_files, hparams, check_files=hparams.check_files, shuffle=False,
                           speaker_ids=trainset.speaker_ids)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset,shuffle=False)#True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = False#True
    
    train_loader = DataLoader(trainset, num_workers=num_workers_, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, train_sampler, trainset


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def warm_start_force_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    pretrained_dict = checkpoint_dict['state_dict']
    model_dict = model.state_dict()
    # Fiter out unneccessary keys
    filtered_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].shape == model_dict[k].shape}
    model_dict_missing = {k: v for k,v in pretrained_dict.items() if k not in model_dict}
    model_dict_mismatching = {k: v for k,v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].shape != model_dict[k].shape}
    pretrained_missing = {k: v for k,v in model_dict.items() if k not in pretrained_dict}
    if model_dict_missing: print(list(model_dict_missing.keys()),'does not exist in the current model and is being ignored')
    if model_dict_mismatching: print(list(model_dict_mismatching.keys()),"is the wrong shape and has been reset")
    if pretrained_missing: print(list(pretrained_missing.keys()),"doesn't have pretrained weights and has been reset")
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    iteration = 0
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    return model, iteration, saved_lookup


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    iteration = checkpoint_dict['iteration']
    iteration = 0
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    return model, iteration, saved_lookup


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    
    #state_dict = {k.replace("encoder_speaker_embedding.weight","encoder.encoder_speaker_embedding.weight"): v for k,v in torch.load(checkpoint_path)['state_dict'].items()}
    #model.load_state_dict(state_dict) # tmp for updating old models
    
    model.load_state_dict(checkpoint_dict['state_dict']) # original
    
    #if 'optimizer' in checkpoint_dict.keys(): optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if 'amp' in checkpoint_dict.keys(): amp.load_state_dict(checkpoint_dict['amp'])
    if 'learning_rate' in checkpoint_dict.keys(): learning_rate = checkpoint_dict['learning_rate']
    #if 'hparams' in checkpoint_dict.keys(): hparams = checkpoint_dict['hparams']
    if 'best_validation_loss' in checkpoint_dict.keys(): best_validation_loss = checkpoint_dict['best_validation_loss']
    if 'average_loss' in checkpoint_dict.keys(): average_loss = checkpoint_dict['average_loss']
    if (start_from_checkpoints_from_zero):
        iteration = 0
    else:
        iteration = checkpoint_dict['iteration']
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration, best_validation_loss, saved_lookup


def save_checkpoint(model, optimizer, learning_rate, iteration, hparams, best_validation_loss, average_loss, speaker_id_lookup, filepath):
    from utils import load_filepaths_and_text
    tqdm.write("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    
    # get speaker names to ID
    speakerlist = load_filepaths_and_text(hparams.speakerlist)
    speaker_name_lookup = {x[1]: speaker_id_lookup[x[2]] for x in speakerlist if x[2] in speaker_id_lookup.keys()}
    
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate,
                #'amp': amp.state_dict(),
                'hparams': hparams,
                'speaker_id_lookup': speaker_id_lookup,
                'speaker_name_lookup': speaker_name_lookup,
                'best_validation_loss': best_validation_loss,
                'average_loss': average_loss}, filepath)
    tqdm.write("Saving Complete")


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, val_teacher_force_till, val_p_teacher_forcing, teacher_force=1):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=num_workers_,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, drop_last=True, collate_fn=collate_fn)
        if teacher_force == 1:
            val_teacher_force_till = 0
            val_p_teacher_forcing = 1.0
        elif teacher_force == 2:
            val_teacher_force_till = 0
            val_p_teacher_forcing = 0.0
        val_loss = 0.0
        diagonality = torch.zeros(1)
        avg_prob = torch.zeros(1)
        for i, batch in tqdm(enumerate(val_loader), desc="Validation", total=len(val_loader), smoothing=0): # i = index, batch = stuff in array[i]
            x, y = model.parse_batch(batch)
            y_pred = model(x, teacher_force_till=val_teacher_force_till, p_teacher_forcing=val_p_teacher_forcing)
            rate, prob = alignment_metric(x, y_pred)
            diagonality += rate
            avg_prob += prob
            loss, gate_loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            # end forloop
        val_loss = val_loss / (i + 1)
        diagonality = (diagonality / (i + 1)).item()
        avg_prob = (avg_prob / (i + 1)).item()
        # end torch.no_grad()
    model.train()
    if rank == 0:
        tqdm.write("Validation loss {}: {:9f}  Average Max Attention: {:9f}".format(iteration, val_loss, avg_prob))
        #logger.log_validation(val_loss, model, y, y_pred, iteration)
        if True:#iteration != 0:
            if teacher_force == 1:
                logger.log_teacher_forced_validation(val_loss, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob)
            elif teacher_force == 2:
                logger.log_infer(val_loss, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob)
            else:
                logger.log_validation(val_loss, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob)
    return val_loss


def calculate_global_mean(data_loader, global_mean_npy, hparams):
    if global_mean_npy and os.path.exists(global_mean_npy):
        global_mean = np.load(global_mean_npy)
        return to_gpu(torch.tensor(global_mean).half()) if hparams.fp16_run else to_gpu(torch.tensor(global_mean).float())
    sums = []
    frames = []
    print('calculating global mean...')
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), smoothing=0.001):
        text_padded, input_lengths, mel_padded, gate_padded,\
            output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states = batch
        # padded values are 0.
        sums.append(mel_padded.double().sum(dim=(0, 2)))
        frames.append(output_lengths.double().sum())
    global_mean = sum(sums) / sum(frames)
    global_mean = to_gpu(global_mean.half()) if hparams.fp16_run else to_gpu(global_mean.float())
    if global_mean_npy:
        np.save(global_mean_npy, global_mean.cpu().numpy())
    return global_mean
