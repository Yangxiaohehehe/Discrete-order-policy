# data/wikitext2/prepare.py
import os
import numpy as np
import tiktoken
from datasets import load_dataset

# 获取当前脚本所在目录
dataset_dir = os.path.dirname(__file__)

# 1. 下载并加载 WikiText-2 (raw 版本保留了真实的大小写和标点)
print("Downloading and loading WikiText-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 2. 将数据集中所有的行拼接成一个长文本
print("Concatenating texts...")
train_data = "\n".join(dataset['train']['text'])
val_data = "\n".join(dataset['validation']['text'])

print(f"Train string length: {len(train_data):,} characters")
print(f"Val string length: {len(val_data):,} characters")

# 3. 使用 GPT-2 BPE 进行分词
print("Encoding with tiktoken (GPT-2 BPE)...")
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# 4. 导出为 nanoGPT 认识的二进制 .bin 文件
print("Exporting to .bin files...")
# 使用 uint16 类型存储，因为 GPT-2 的词表大小是 50257，刚好在 uint16 (0~65535) 范围内
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(dataset_dir, 'train.bin'))
val_ids.tofile(os.path.join(dataset_dir, 'val.bin'))

print("WikiText-2 preparation completed!")