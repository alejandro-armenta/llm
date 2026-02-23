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

print("Embeddings", embeddings.shape)

#context_aware embeddings

d_model = 768

d_k = 64

W_q = nn.Linear(d_model,d_k,bias=False)
W_k = nn.Linear(d_model,d_k,bias=False)
W_v = nn.Linear(d_model,d_k,bias=False)

#estos vectores van a apuntar a lugares diferentes y se van a alinear
Q = W_q(embeddings)

K = W_k(embeddings)

token * token en sequencia
a = Q @ K.transpose(-2,-1)

#noramlizando cada score
a = a / math.sqrt(d_k)

attn_weights = F.softmax(a, dim=-1)

each query
print(attn_weights[0,0].sum())