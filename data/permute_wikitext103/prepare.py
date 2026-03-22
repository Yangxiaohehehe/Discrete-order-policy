import os
import numpy as np

# 配置参数
dataset_name = 'wikitext103' # 你想要打乱的数据集
block_size = 256 # 必须与你训练时的 block_size 严格保持一致
seed = 42 # 固定随机种子，确保 train 和 val 使用完全相同的打乱规则

data_dir = os.path.join('data', dataset_name)
out_dir = os.path.join('data', f'{dataset_name}_permuted')

os.makedirs(out_dir, exist_ok=True)

# 1. 生成全局固定的打乱索引 (Fixed Permutation Indices)
np.random.seed(seed)
fixed_perm = np.random.permutation(block_size)
print(f"Generated fixed permutation for block_size {block_size}.")
print(f"First 10 indices mapping: {fixed_perm[:]}")

# 2. 处理 train 和 val 分野
for split in ['train', 'val']:
    in_path = os.path.join(data_dir, f'{split}.bin')
    out_path = os.path.join(out_dir, f'{split}.bin')
    
    if not os.path.exists(in_path):
        print(f"Skipping {split}, file not found.")
        continue
        
    # 以只读模式加载原始二进制数据
    data = np.memmap(in_path, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    
    # 截断尾部多余的 token，使其能被 block_size 完美整除
    num_blocks = total_tokens // block_size
    valid_length = num_blocks * block_size
    data_truncated = data[:valid_length]
    
    print(f"\nProcessing {split}...")
    print(f"Original length: {total_tokens:,}, Truncated to: {valid_length:,} ({num_blocks:,} blocks)")
    
    # 【核心矩阵运算】：利用 Numpy 的高级索引进行极速全局打乱
    # 把一维数据 reshape 成 (num_blocks, 256) 的矩阵
    data_reshaped = data_truncated.reshape(num_blocks, block_size)
    # 对矩阵的每一行（列维度）施加固定的 fixed_perm 索引
    data_permuted = data_reshaped[:, fixed_perm]
    # 重新展平为一维数组
    data_permuted_flat = data_permuted.flatten()
    
    # 写入新的二进制文件
    out_memmap = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(valid_length,))
    out_memmap[:] = data_permuted_flat[:]
    out_memmap.flush() # 强制写入硬盘
    
    print(f"Saved permuted {split} data to {out_path}")

print("\n🎉 Fixed permutation dataset created successfully!")