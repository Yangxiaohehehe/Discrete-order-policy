# AR training with standard validation loss only.

out_dir = 'out-wikitext103-ar-permute'
eval_interval = 500
eval_iters = 200
log_interval = 50

wandb_log = True
wandb_project = 'ao-gpt-experiments-256'
wandb_run_name = 'wikitext103-ar-permute-big'

dataset = 'wikitext103'
batch_size = 32
block_size = 256
gradient_accumulation_steps = 4
permute_data = True
permute_seed = 42

model_type = 'aogpt'
aogpt_train_mode = 'AR'
main_eval_mode = 'AR'
generalization_eval_mode = ''
n_layer = 6
n_head = 6
n_embd = 576
dropout = 0

learning_rate = 6e-4
max_iters = 7000
lr_decay_iters = 7000
min_lr = 6e-5
beta2 = 0.99
warmup_iters = 0
