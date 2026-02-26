import torch
import torch.nn.functional as F

from gptconfig import GPTConfig, GPT

from transformers import AutoTokenizer


config = GPTConfig()
model = GPT(config=config)

def load_gpt2_weights(model):

    from transformers import GPT2LMHeadModel

    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')

    hf_state = hf_model.state_dict()

    our_state = model.state_dict()

    #print(our_state is model.state_dict())

    # Copy embeddings
    our_state['token_embed.weight'].copy_(hf_state['transformer.wte.weight'])
    our_state['pos_embed.weight'].copy_(hf_state['transformer.wpe.weight'])
    
    # Copy each Transformer block
    for i in range(model.config.num_layers):
        
        # Layer norms
        our_state[f'blocks.{i}.ln1.weight'].copy_(hf_state[f'transformer.h.{i}.ln_1.weight'])
        our_state[f'blocks.{i}.ln1.bias'].copy_(hf_state[f'transformer.h.{i}.ln_1.bias'])
        our_state[f'blocks.{i}.ln2.weight'].copy_(hf_state[f'transformer.h.{i}.ln_2.weight'])
        our_state[f'blocks.{i}.ln2.bias'].copy_(hf_state[f'transformer.h.{i}.ln_2.bias'])
        
        # Attention (need to transpose!)
        our_state[f'blocks.{i}.attn.qkv_proj.weight'].copy_(hf_state[f'transformer.h.{i}.attn.c_attn.weight'].T)
        our_state[f'blocks.{i}.attn.out_proj.weight'].copy_(hf_state[f'transformer.h.{i}.attn.c_proj.weight'].T)
        
        # FFN (need to transpose!)
        our_state[f'blocks.{i}.ffn.fc1.weight'].copy_(hf_state[f'transformer.h.{i}.mlp.c_fc.weight'].T)
        our_state[f'blocks.{i}.ffn.fc1.bias'].copy_(hf_state[f'transformer.h.{i}.mlp.c_fc.bias'])
        our_state[f'blocks.{i}.ffn.fc2.weight'].copy_(hf_state[f'transformer.h.{i}.mlp.c_proj.weight'].T)
        our_state[f'blocks.{i}.ffn.fc2.bias'].copy_(hf_state[f'transformer.h.{i}.mlp.c_proj.bias'])


    # Final layer norm
    our_state['ln_f.weight'].copy_(hf_state['transformer.ln_f.weight'])
    our_state['ln_f.bias'].copy_(hf_state['transformer.ln_f.bias'])

    #model.load_state_dict(our_state)

    return model


model = load_gpt2_weights(model=model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=512, temperature=1.0):
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

@torch.no_grad()
def generate_simple(model, tokenizer, prompt, max_new_tokens=20):
    """Generate text using greedy decoding (always pick most likely token).Args:model: Our MiniGPT with loaded weightstokenizer: GPT-2 tokenizerprompt: Starting textmax_new_tokens: How many new tokens to generateReturns:Generated text string"""
    model.eval()
    device= next(model.parameters()).device
    
    # Encode prompt
    token_ids= tokenizer.encode(prompt, return_tensors="pt").to(device)
    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Get logits for current sequence
        logits= model(token_ids)
        
        # Get the last position's logits
        next_logits= logits[:,-1, :]
        # (batch, vocab_size)
        # Greedy: pick the highest-scoring token
        next_token= next_logits.argmax(dim=-1, keepdim=True)  
        # (batch, 1)
        # Append to sequence
        token_ids= torch.cat([token_ids, next_token], dim=1)
        
        # Stop if we generate end-of-text token
        if next_token.item() == tokenizer.eos_token_id:
            break 
    
    # Decode back to text
    return tokenizer.decode(token_ids[0])


tokenizer = AutoTokenizer.from_pretrained('gpt2')
#tokenizer.pad_token = tokenizer.eos_token

prompt= "The quick brown fox"

generated = generate(model, tokenizer, prompt, max_new_tokens=30)

print(generated)