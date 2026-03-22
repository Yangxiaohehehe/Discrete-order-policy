# config/train_wikitext2_aogpt.py

# I/O
out_dir = 'out-wikitext2-aogpt'
eval_interval = 250  # 数据量大了，可以每 500 步评估一次
eval_iters = 200
log_interval = 10

# WandB 设置
wandb_log = True
wandb_project = 'ao-gpt-experiments'
wandb_run_name = 'Random-small-wikitext2'

# 数据集切换为我们刚刚生成的 wikitext2
dataset = 'wikitext2'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # 256个 BPE token 相当于很长的一段话了

# 模型架构 (保持之前的 Baby GPT 大小)
n_layer = 2
n_head = 4
n_embd = 128
dropout = 0.1 # 加入轻微的 Dropout 防止模型在 200万 Token 上过拟合

# 优化器设置
learning_rate = 1e-3
max_iters = 5000     # 数据多了，训练步数稍微拉长一点
lr_decay_iters = 5000 
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 200    # 预热也稍微给多一点点