
from Tokenizer import WordTokenizer

tokenizer = WordTokenizer(max_vocab_size=10)

training_text = """
The cat sat on the mat.
The cat was on the mat.
The dog sat on the mat.
"""

tokenizer.fit(training_text)

print(f"Vocabulary:{tokenizer.word_to_id}")

text1 = "the cat sat"
ids1= tokenizer.encode(text1)
print(f"\n'{text1}' → {ids1}")

print(f"Decoded: '{tokenizer.decode(ids1)}'")

text2= "the elephant sat" # "elephant" not in vocab!
ids2= tokenizer.encode(text2)
print(f"\n'{text2}' → {ids2}")

print(f"Decoded: '{tokenizer.decode(ids2)}'")

ids3 = tokenizer.encode("the cat", add_special_tokens=True)
print(f"\nWith special tokens:{ids3}")

print(f"Decoded (showing special): '{tokenizer.decode(ids3, skip_special_tokens=False)}'")

print(f"Decoded (hiding special): '{tokenizer.decode(ids3, skip_special_tokens=True)}'")
