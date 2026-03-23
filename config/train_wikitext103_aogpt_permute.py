# config/train_wikitext103_aogpt.py

# 实验名称与路径
out_dir = 'out-wikitext103-aogpt-permute'
eval_interval = 500 # 步数拉长，每 1000 步评估一次即可
eval_iters = 200
log_interval = 50

# WandB 记录
wandb_log = True
wandb_project = 'ao-gpt-experiments-256'
#wandb_run_name = 'R_w103-loss-R-data-l2r-small_block-nodrop'
wandb_run_name = 'R-data-Per-loss-AR'

# 数据集设定
dataset = 'permute_wikitext103'
batch_size = 32
block_size = 256 # 上下文长度依然保持 256 个 BPE Token
gradient_accumulation_steps = 4 # 累加梯度，模拟 32 * 4 = 128 的全局 Batch Size，让 Loss 更平滑
permute_data = True
permute_seed = 42

# 模型规模 (Baby GPT)
model_type = 'aogpt'
aogpt_train_mode = 'Random'
eval_modes = ['Random'] # 单评估
#eval_modes = ['random', 'ar']
n_layer = 2
n_head = 4
n_embd = 128
dropout = 0 # 数据量大了，dropout 稍微调低，允许模型尽情吸收知识

# 优化器设置
learning_rate = 6e-4 # 对于大一点的数据集和累加 Batch，峰值学习率稍微降一点，求稳
max_iters = 7000    # 总步数大幅增加到 50,000 步
lr_decay_iters = 7000 
min_lr = 6e-5
beta2 = 0.99
warmup_iters = 0  # 预热期也相应拉长，让模型在乱序海洋中缓慢苏醒