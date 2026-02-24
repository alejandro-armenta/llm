import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from llm import TransformerBlock

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1

    def __post_init__(self):
        assert self.embed_dim % self.num_heads == 0, f"embed_dim ({self.embed_dim}) must divide by num heads ({self.num_heads})"


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        
        #solo es una tabla con vectores
        #aqui se aprenden los embeddings

        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        self.pos_embed = nn.Embedding(config.max_seq_len, config.embed_dim)

        self.dropout = nn.Dropout(config.dropout)

        #12 modulos
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.embed_dim,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout
                )
                
                for i in range(config.num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(config.embed_dim)

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

        
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.2)
        nn.init.normal_(self.pos_embed.weight, std=0.2)


    def forward(self, token_ids, return_attention=False):
        
        batch, seq = token_ids.shape
        device = token_ids.device

        tok_emb = self.token_embed(token_ids)

        print(tok_emb.shape)

        positions = torch.arange(seq, device=device)
        
        print(positions.shape)

        #16 * 768
        pos_emb = self.pos_embed(positions)

        #tienen los mismos vectores se repiten por batch
        x = self.dropout(tok_emb + pos_emb)
        
        mask = torch.tril(torch.ones(seq,seq, device=device))
        
        attention_weights = []
        for b in self.blocks:
            x, attn = b(x, mask)

            if return_attention:
                attention_weights.append(attn)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        if return_attention:
            return logits, attention_weights

        return logits
    
config = GPTConfig()
a = GPT(config=config)

batch_size = 2
seq_len = 16
token_ids = torch.randint(0,config.vocab_size, (batch_size, seq_len))
print(token_ids)
a(token_ids)

