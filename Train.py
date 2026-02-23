

#with open('shakespeare.txt', 'r') as f:
#    text = f.read()
#    print(len(text))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

#we need subwords in words not around them 

tokenizer = Tokenizer(BPE(unk_token='<UNK>'))

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=1000, 
           min_frequency=2,
           special_tokens=['<PAD>',
                           '<UNK>',
                           '<BOS>',
                           '<EOS>',
                           ]
                           )

tokenizer.train(files=['shakespeare.txt'], trainer=trainer)

print(tokenizer.get_vocab_size())

tokenizer.save('shakespeare_bpe.json')