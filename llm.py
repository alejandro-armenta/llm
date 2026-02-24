import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Attention import multiheadattention, create_causal_mask

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # si tienen bias
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

ff = FeedForward(d_model=768, d_ff=3072, dropout=0.1)

embeddings_test = torch.randn(2,6,768)

ff(embeddings_test)

ln = nn.LayerNorm(normalized_shape=768)

x_ln= torch.randn(2, 6, 768)

output_ln = ln(x_ln)

#print("orig mean", x_ln.mean())
#print("output mean", output_ln.mean())
#print("orig std", x_ln.std())
#print("output std", output_ln.std())


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):

        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape=d_model)
        self.ln2 = nn.LayerNorm(normalized_shape=d_model)

        self.attn = multiheadattention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        #aqui hay un skip connection
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)

        #aqui hay un skip connection
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x, attn_weights


tb = TransformerBlock(d_model=768, num_heads=12, d_ff=3072, dropout=0.1)
embeddings_block= torch.randn(2, 6, 768)
mask_block = create_causal_mask(6)

output_block, attn_weights_block = tb(embeddings_block, mask_block)



