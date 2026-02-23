import tiktoken

with open('shakespeare.txt', 'r') as f:
    text = f.read()
    pass

#print(text)
enc = tiktoken.get_encoding('cl100k_base')
#print(enc)
tokens = enc.encode(text)
print(tokens)

#768 * sequence 

#los embeddings son Dense vectors

#each token gets the same vector regardless of context.

#seq, embedding dimension
#(6,  768)
#rows of vectors 
#each row is a token

