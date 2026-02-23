import torch
import torch.nn as nn
import torch.nn.functional as F
import math

batch_size = 2
#esta viendo a todos estos al mismo tiempo
# no ve a la otra batch
seq_len = 6
embed_dim = 768

#2 * 6 * 768

#static embeddings

embeddings = torch.randn(batch_size, seq_len, embed_dim)

#print("Embeddings", embeddings.shape)

#context_aware embeddings

d_model = 768

d_k = 64

#esta cosa slo transforma la ultima dimension
W_q = nn.Linear(d_model,d_k,bias=False)
W_k = nn.Linear(d_model,d_k,bias=False)
W_v = nn.Linear(d_model,d_k,bias=False)

#estos vectores van a apuntar a lugares diferentes y se van a alinear


Q = W_q(embeddings)

K = W_k(embeddings)


#token * token en sequencia
"""
        keys
queries 
"""

a = Q @ K.transpose(-2,-1)

#noramlizando cada score
a = a / math.sqrt(d_k)

attn_weights = F.softmax(a, dim=-1)

"""
#2*6*6            2 * 6 * 64
#6 * 64
----probs-----   ----v-------
|
|
6|              6
|
|

"""

V = W_v(embeddings)

#aqui accedes al valor de los libros
#si tienen que ser las queries porque 

"""
        values
queries 
"""

output = attn_weights @ V

#print(output.shape)

def scaled_dot_product_attention(Q,K,V, mask=None):
    d_k = Q.size(-1)

    a = Q @ K.transpose(-2,-1)

    a = a / math.sqrt(d_k)

    if mask is not None:
        a = a.masked_fill(mask == 0, float('-inf'))
        #print(a)

    #tiene que aprender a usar lo que tiene para predecir la siguiente palabra no mas!
    #los -inf no cuentan para calcular pesos
    #entonces toda la atencion va hacia el pasado

    attn_weights = F.softmax(a, dim=-1)

    #print(attn_weights[0].sum(dim=-1))

    output = attn_weights @ V

    return output, attn_weights
    

#esta mascara es importante porque si no no funciona
def create_causal_mask(seq_len):

    mask = torch.tril(torch.ones(seq_len, seq_len))

    return mask

mask = create_causal_mask(6)
output, attn_weights = scaled_dot_product_attention(Q,K,V,mask=mask)

#print(attn_weights)


#each head sees all tokens but learns different patterns

class multiheadattention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        #768 a 2304
        self.qkv_proj = nn.Linear(d_model, 3*d_model,bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, mask=None):
        
        batch, seq, d_model = x.shape #768

        qkv = self.qkv_proj(x)

                                                #este se hizo mas grande y si se armo
        qkv = qkv.reshape(batch, seq, 3, self.num_heads, self.d_head)

        qkv = qkv.permute(2,0,3,1,4)

        Q = qkv[0]
        K = qkv[1]
        V = qkv[2]

        d_k = self.d_head

        a = Q @ K.transpose(-2,-1)

        a = a / math.sqrt(d_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)

            a = a.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(a, dim=-1)

        attn_weights = self.dropout(attn_weights)

        output = attn_weights @ V

        output = output.transpose(1,2)

        output = output.reshape(batch, seq, d_model)

        o = self.out_proj(output)

        return o, attn_weights

a = multiheadattention(d_model=768, num_heads=6, dropout=0.1)

#aqui estan los numheads
embeddings_test = torch.randn(2,6,768)
mask = create_causal_mask(6)

o, a = a(embeddings_test, mask)

print(embeddings_test.shape)
print(o.shape)
print(a.shape)