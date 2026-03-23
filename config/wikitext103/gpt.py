# Original GPT baseline on WikiText103.

out_dir = 'out-wikitext103-gpt'
eval_interval = 500
eval_iters = 200
log_interval = 50

wandb_log = True
wandb_project = 'ao-gpt-experiments-128'
wandb_run_name = 'wikitext103-gpt'

dataset = 'wikitext103'
batch_size = 64
block_size = 128
gradient_accumulation_steps = 4
permute_data = False
permute_seed = 42

model_type = 'gpt'
aogpt_train_mode = 'AR'
main_eval_mode = 'AR'
generalization_eval_mode = ''
n_layer = 2
n_head = 4
n_embd = 128
dropout = 0

learning_rate = 6e-4
max_iters = 7000
lr_decay_iters = 7000
min_lr = 6e-5
beta2 = 0.99
warmup_iters = 0
