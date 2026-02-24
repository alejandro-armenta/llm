from transformers import AutoTokenizer

from functools import partial

from torch.utils.data import Dataset, DataLoader


with open('shakespeare.txt', 'r') as f:
    text = f.read()

class TextDataset(Dataset):
    def __init__(self, text, chunk_size=128):
        self.chunks = []
        for i in range(0, len(text) - chunk_size, chunk_size):
            self.chunks.append(text[i:i+chunk_size])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        return self.chunks[index]


dataset = TextDataset(text=text, chunk_size=256)
#print(len(dataset))
#print(dataset[1])

tokenizer = AutoTokenizer.from_pretrained('gpt2')

#el bactch es un grupo de chunks

def collate_fn(batch, tokenizer, max_length=128):
    #batch * arbitrary

    #batch * max_length

    encoded = tokenizer(
        batch, 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors='pt'
        )
    
    return encoded['input_ids'], encoded['attention_mask']

#es un objeto que le pasa esos argumentos dejando el primero libre
collate = partial(collate_fn, tokenizer=tokenizer, max_length=128)

print(collate)



#DataLoader()