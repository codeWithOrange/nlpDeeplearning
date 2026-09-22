text = "I love machine learning"
tokens = text.lower().split()
vocab = {word: i + 1 for i, word in enumerate(set(tokens))}
sequence = [vocab[word] for word in tokens]
print(tokens)
print(vocab)
print(sequence)
# practical with keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

texts = [
    "I love machine learning",
    "I love deep learning",
    "machine learning is powerful",
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequence = tokenizer.texts_to_sequences(text)
print(sequence)
padded = pad_sequences(sequence, padding="post")
print(padded)
