from transformers import AutoTokenizer

import torch
import torch.nn.functional as F

from gptconfig import GPTConfig, GPT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

chech = torch.load('best_model_big.pt', map_location=device)

print(chech['val_loss'])


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


model.to(device)

model.load_state_dict(chech['model_state_dict'])

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


@torch.no_grad()
def generate_(model, tokenizer, prompt, max_new_tokens=30):
    model.eval()

    device = next(model.parameters()).device

    token_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    #print(token_ids)

    for _ in range(max_new_tokens):
        logits = model(token_ids)

        next_logits = logits[:,-1,:]

        indices = next_logits.argmax(dim=-1, keepdim=True)

        token_ids = torch.cat([token_ids, indices], dim=-1)

        if indices.item() == tokenizer.eos_token_id:
            break


    return tokenizer.decode(token_ids[0])


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=30, temperature=1.0):
    """Generate text with temperature control."""
    model.eval()
    device = next(model.parameters()).device

    token_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        logits = model(token_ids)
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_ids = torch.cat([token_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(token_ids[0])



prompts = [
    "The king",
    "To be or not to be",
    "Friends, Romans, countrymen",
    "All the world's a stage"
]
    
for prompt in prompts:
    output = generate(model=model, tokenizer=tokenizer, prompt=prompt)
    print(output)

