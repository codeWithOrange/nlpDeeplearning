import numpy as np

text = "the quick brown fox jumps over the lazy dog"
words = text.split()
print("words", words)


vocab = sorted(set(words))
vocab_size = len(words)


word_to_id = {}
id_to_word = {}
for i, w in enumerate(vocab):
    word_to_id[w] = i
    id_to_word[i] = w

print("vocabulary: ", vocab)
print("vocabulary size: ", vocab_size)
print("word -> ID: ", word_to_id)
print("ID -> word: ", id_to_word)

target = "lazy"

print(word_to_id[target])
print(id_to_word[2])


def one_hot(word):
    vector = np.zeros(vocab_size)
    vector[word_to_id[word]] = 1
    return vector


print("one hot vector for 'fox' ")

print(one_hot("fox"))


window = 2
cbow_examples = []

for i in range(len(words)):
    target = words[i]
    context = []
    for j in range(window, window + 1):
        if j == 0:
            continue
        neighbor_index = i + j
        if 0 <= neighbor_index < len(words):
            context.append(words[neighbor_index])
    if len(context) > 0:
        cbow_examples.append((context, target))
print("number of cbow traiing examples: ", len(cbow_examples))
print()
print("first 3 examples (context->target): ")
for context, target in cbow_examples[:10]:
    print(f"{context} -> {target}")
