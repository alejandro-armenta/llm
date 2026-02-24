from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

from gptconfig import GPTConfig, GPT

import torch

model = GPT(GPTConfig(num_layers=2, embed_dim=256, num_heads=4, d_ff=1024))

model.eval()

prompt = "The quick brown fox"
token_ids = tokenizer.encode(text=prompt, return_tensors='pt')

print(prompt)

with torch.no_grad():
    logits = model(token_ids)

last_logits = logits[0,-1, :]

values, indices = torch.softmax(last_logits, dim=-1).topk(5)

for prob, idx in zip(values, indices):
    token = tokenizer.decode([idx])
    print(f'{prob:.4f} {token}')

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


