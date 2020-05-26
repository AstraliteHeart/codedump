current_iteration = iteration
decay_start = 180000
A_ = 20e-5
B_ = 30000
C_ = 0e-5
min_learning_rate = 10e-5
epochs_between_updates = 1
drop_frame_rate = min(0.000010 * max(current_iteration-5000,0), 0.2) # linearly increase DFR from 0.0 to 0.25 from iteration 5000 to 55000.
p_teacher_forcing = 0.99
teacher_force_till = 0
val_p_teacher_forcing=0.80
val_teacher_force_till=30
grad_clip_thresh = 1.0
