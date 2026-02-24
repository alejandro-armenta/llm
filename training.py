from transformers import AutoTokenizer

from functools import partial

import torch

from torch.utils.data import random_split, DataLoader

from utils import TextDataset, collate_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

def set_seed(seed=42):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

set_seed(42)

with open('shakespeare.txt', 'r') as f:
    text = f.read()



tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

dataset = TextDataset(text=text, chunk_size=256)


train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset, 
    [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
    )

collate = partial(collate_fn, tokenizer=tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)

input_ids, attn = next(iter(train_loader))

print(input_ids.shape)
print(attn.shape)