"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from AOGPT import AOGPTConfig, AOGPT

os.environ["WANDB_API_KEY"] = "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"
os.environ["WANDB_MODE"] = "online"
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True 
wandb_project = 'ao-gpt-experiments' # 你的项目名称
wandb_run_name = 'mdm_random_order_run1' # 你的实验运行名称
# data
dataset = 'openwebtext'
permute_data = False
permute_seed = 42
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
model_type = 'aogpt'          # 可选: 'gpt' (原生 NanoGPT) 或 'aogpt' (AO-GPT)
aogpt_train_mode = 'AR' # 仅在 model_type == 'aogpt' 时生效
main_eval_mode = 'Random' # AOGPT 主评估模式
generalization_eval_mode = '' # 额外泛化评估模式；例如 Random 训练时可设为 'AR'
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
val_token_loss_log = True
val_token_loss_batch_size = 8
val_token_loss_seed = 12345
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
config['main_eval_mode'] = main_eval_mode
config['generalization_eval_mode'] = generalization_eval_mode
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

np.random.seed(permute_seed)
fixed_perm = torch.tensor(np.random.permutation(block_size), dtype=torch.long) if permute_data else None
fixed_val_batch_size = min(val_token_loss_batch_size, batch_size)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    if permute_data:
        perm_idx = fixed_perm.to(device)
        x = x[:, perm_idx]
        y = y[:, perm_idx]
    return x, y

def get_fixed_batch(split, fixed_batch_size, fixed_seed):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    rng = np.random.default_rng(fixed_seed)
    ix = rng.integers(0, len(data) - block_size, size=fixed_batch_size)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    if permute_data:
        perm_idx = fixed_perm.to(device)
        x = x[:, perm_idx]
        y = y[:, perm_idx]
    return x, y

fixed_val_x, fixed_val_y = get_fixed_batch('val', fixed_val_batch_size, val_token_loss_seed)
val_token_loss_history = []
val_token_loss_steps = []

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    if model_type == 'gpt':
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif model_type == 'aogpt':
        gptconf = AOGPTConfig(**model_args)
        model = AOGPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    if model_type == 'gpt':
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif model_type == 'aogpt':
        gptconf = AOGPTConfig(**model_args)
        model = AOGPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# elif init_from.startswith('gpt2'):
#     print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
#     # initialize from OpenAI GPT-2 weights
#     override_args = dict(dropout=dropout)
#     model = GPT.from_pretrained(init_from, override_args)
#     # read off the created config params, so we can store them into checkpoint correctly
#     for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#         model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        if model_type == 'gpt':
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        elif model_type == 'aogpt':
            losses = torch.zeros(eval_iters)
            if generalization_eval_mode:
                generalization_losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, mode=main_eval_mode)
                    if generalization_eval_mode:
                        _, generalization_loss = model(X, mode=generalization_eval_mode)
                losses[k] = loss.item()
                if generalization_eval_mode:
                    generalization_losses[k] = generalization_loss.item()
            out[split] = losses.mean().item()
            if generalization_eval_mode:
                metric_suffix = generalization_eval_mode.lower()
                out[f"{split}_generalization_{metric_suffix}"] = generalization_losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def estimate_fixed_val_token_loss():
    model.eval()
    with ctx:
        if model_type == 'gpt':
            logits, _ = model(fixed_val_x, fixed_val_y)
            token_losses = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                fixed_val_y.view(-1),
                ignore_index=-1,
                reduction='none',
            ).view(fixed_val_y.size(0), fixed_val_y.size(1))
        else:
            # For Random training we visualize only AR-mode validation behavior.
            logits, _ = model(fixed_val_x, mode='AR')
            token_losses = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                fixed_val_x.reshape(-1),
                ignore_index=-1,
                reduction='none',
            ).view(fixed_val_x.size(0), fixed_val_x.size(1))
    model.train()
    return token_losses.mean(dim=0).float().cpu().numpy()

def build_val_token_loss_figure(history, steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    history_np = np.asarray(history, dtype=np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].plot(np.arange(history_np.shape[1]), history_np[-1], linewidth=1.5)
    axes[0].set_title('Fixed Val Batch Mean Per-Token Loss')
    axes[0].set_xlabel('Token Position')
    axes[0].set_ylabel('Loss')

    im = axes[1].imshow(history_np, aspect='auto', origin='lower', interpolation='nearest')
    axes[1].set_title('Per-Token Loss Over Eval Steps')
    axes[1].set_xlabel('Token Position')
    axes[1].set_ylabel('Eval Step')
    axes[1].set_yticks(np.arange(len(steps)))
    axes[1].set_yticklabels([str(step) for step in steps])
    fig.colorbar(im, ax=axes[1], fraction=0.025, pad=0.02, label='Loss')

    return fig

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if model_type == 'aogpt' and generalization_eval_mode:
            generalization_key = generalization_eval_mode.lower()
            print(
                f"  generalization_{generalization_key}_loss: "
                f"train {losses[f'train_generalization_{generalization_key}']:.4f}, "
                f"val {losses[f'val_generalization_{generalization_key}']:.4f}"
            )
        if wandb_log:
            log_payload = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }
            if model_type == 'aogpt' and generalization_eval_mode:
                generalization_key = generalization_eval_mode.lower()
                log_payload[f"train/generalization_{generalization_key}_loss"] = losses[f"train_generalization_{generalization_key}"]
                log_payload[f"val/generalization_{generalization_key}_loss"] = losses[f"val_generalization_{generalization_key}"]
            if val_token_loss_log:
                token_loss_curve = estimate_fixed_val_token_loss()
                val_token_loss_history.append(token_loss_curve)
                val_token_loss_steps.append(iter_num)
                figure = build_val_token_loss_figure(val_token_loss_history, val_token_loss_steps)
                metric_prefix = "val/generalization_ar" if model_type == 'aogpt' and aogpt_train_mode == 'Random' else "val/main"
                log_payload[f"{metric_prefix}_fixed_batch_token_loss_mean"] = float(token_loss_curve.mean())
                log_payload[f"{metric_prefix}_fixed_batch_token_loss_plot"] = wandb.Image(figure)
                import matplotlib.pyplot as plt
                plt.close(figure)
            wandb.log(log_payload)
        val_loss_for_ckpt = losses['val']
        if val_loss_for_ckpt < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss_for_ckpt
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if model_type == 'gpt':
                logits, loss = model(X, Y)
            elif model_type == 'aogpt':
                logits, loss = model(X, mode=aogpt_train_mode)
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
