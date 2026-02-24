from torch.utils.data import Dataset, DataLoader, random_split

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
        truncation=True, 
        max_length=max_length, 
        return_tensors='pt'
        )
    
    return encoded['input_ids'], encoded['attention_mask']
