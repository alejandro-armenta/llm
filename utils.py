import torch

from torch.utils.data import Dataset

import torch.nn.functional as F

from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, text, chunk_size=128):
        self.chunks = []
        for i in range(0, len(text) - chunk_size, chunk_size):
            self.chunks.append(text[i:i+chunk_size])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        return self.chunks[index]


def collate_fn(batch, tokenizer, max_length=128):

    encoded = tokenizer(
        batch, 
        padding=True, 
        #padding='max_length', 
        truncation=True, 
        max_length=max_length, 
        return_tensors='pt'
        )
    
    return encoded['input_ids'], encoded['attention_mask']


def train_epoch(model, dataloader, optimizer, scheduler, tokenizer, device, clip_norm=1.0):

    model.train()

    progress = tqdm(dataloader, desc='training epoch')

    epoch_loss = 0

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

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
