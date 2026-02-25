from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from functools import partial


import torch

import torch.nn.functional as F

from torch.optim import AdamW

from torch.utils.data import random_split, DataLoader


from utils import TextDataset, collate_fn

from gptconfig import GPTConfig, GPT

from tqdm import tqdm

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

collate = partial(
    collate_fn, 
    tokenizer=tokenizer, 
    max_length=128)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)

"""
input_ids, attn_mask = next(iter(train_loader))

#ahora es el mayor de todos truncado
print(input_ids.shape)
print(attn)

zero gradients each batch

Forward

Compute Loss

Backward pass

update weights
"""

config = GPTConfig(
    vocab_size=50257,
    max_seq_len=128,
    embed_dim=256,
    num_heads=4,
    num_layers=4,
    d_ff=1024,
    dropout=0.1
)

model = GPT(config=config)

model.to(device=device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

learning_rate = 3e-4
num_warmup_steps = 100
num_epochs = 3

total_steps = len(train_loader) * num_epochs

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

def train_epoch(model, dataloader, optimizer, device, clip_norm=1.0):

    model.train()

    progress = tqdm(dataloader, desc='training epoch')

    for input_ids, attn_mask in progress:
        
        input_ids = input_ids.to(device)

        inputs = input_ids[:,:-1]

        targets = input_ids[:,1:]

        optimizer.zero_grad(set_to_none=True)
        
        logits = model(inputs)

        a = logits.view(-1, logits.size(-1))
        
        b = targets.reshape(-1)

        loss = F.cross_entropy(a, b, ignore_index=tokenizer.pad_token_id)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()

        scheduler.step()

        progress.set_postfix(loss=f"{loss.item():.4f}")

        




train_epoch(model=model, dataloader=train_loader, optimizer=optimizer, device=device)