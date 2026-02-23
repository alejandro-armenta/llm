class WordTokenizer:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word_to_id = {}
        self.id_to_word = {}

    def fit(self, text):
        import re

        words = re.findall(r"\b\w+\b", text.lower())

        from collections import Counter
        counts = Counter(words)

        self.word_to_id = {
            '<PAD>' : 0,
            '<UNK>' : 1,
            '<BOS>' : 2,
            '<EOS>' : 3
        }

        self.id_to_word = {v:k for k,v in self.word_to_id.items()}

        top_words = counts.most_common(self.max_vocab_size - 4)

        for i, (word, count) in enumerate(top_words, start=4):
            self.word_to_id[word] = i
            self.id_to_word[i] = word


    def encode(self, text, add_special_tokens = False):
        
        words = text.lower().split()
        
        ids = [self.word_to_id.get(w,1) for w in words]

        if add_special_tokens:
            ids = [2] + ids + [3]
            
        return ids
        
        
    def decode(self, ids, skip_special_tokens=True):

        words = []
        
        for i in ids:
            word = self.id_to_word[i]
        
            if skip_special_tokens and word in [
                '<PAD>',
                '<BOS>',
                '<EOS>'
                ]:
                continue
        
            words.append(word)
        
        
        return " ".join(words)
    
    @property
    def vocab_size(self):
        return len(self.word_to_id)


