from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from functools import partial

import torch

import torch.nn.functional as F

from torch.optim import AdamW

from torch.utils.data import random_split, DataLoader

from utils import TextDataset, collate_fn, train_epoch, evaluate

from gptconfig import GPTConfig, GPT

import matplotlib.pyplot as plt

import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cpu')

print(device)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

def set_seed(seed=42):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

set_seed(42)


def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    # Parse each line as a JSON object
                    json_object = json.loads(line)
                    data.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
    return data

file_path_train_data = 'data/small-117M.train.jsonl'
file_path_valid_data = 'data/small-117M.valid.jsonl'

train_data = read_jsonl_file(file_path_train_data)
valid_data = read_jsonl_file(file_path_valid_data)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

train_dataset = TextDataset(data=train_data)
valid_dataset = TextDataset(data=valid_data)

print(train_dataset)

collate = partial(
    collate_fn, 
    tokenizer=tokenizer, 
    max_length=1024)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate, num_workers=10)

item, _ = next(iter(train_loader))
print(item.shape)

val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate)

item, _ = next(iter(val_loader))
print(item.shape)


config = GPTConfig()
model = GPT(config=config)

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
"""

model.to(device=device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

learning_rate = 3e-4

num_warmup_steps = 100

num_epochs = 3

total_steps = len(train_loader) * num_epochs

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)


best_val_loss = float('inf')

train_losses = []
val_losses = []

for e in range(num_epochs):

    train_loss = train_epoch(model=model, dataloader=train_loader, optimizer=optimizer, scheduler=scheduler, tokenizer=tokenizer, device=device, clip_norm=1.0)
    train_losses.append(train_loss)

    val_loss = evaluate(model=model, dataloader=val_loader, tokenizer=tokenizer, device=device)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch':e,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            'train_loss':train_loss,
            'val_loss':val_loss
        }, f='best_model.pt')

print(best_val_loss)



plt.figure(figsize=(8,5))
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
plt.plot(epochs, val_losses, 'r-o', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss.png')
#plt.show()


