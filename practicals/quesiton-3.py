"""
Train CBOW and Skip-Gram Word2Vec models on a sample text corpus, compute word similarities
and analogies, and compare the performance of both models
"""

from gensim.models import Word2Vec
import time

sentences = [
    ["king", "is", "a", "man"],
    ["queen", "is", "a", "woman"],
    ["king", "rules", "the", "kingdom"],
    ["queen", "rules", "the", "kingdom"],
    ["man", "is", "strong"],
    ["woman", "is", "strong"],
    ["boy", "is", "young"],
    ["girl", "is", "young"],
    ["cat", "eats", "fish"],
    ["dog", "eats", "meat"],
    ["cat", "drinks", "milk"],
    ["dog", "drinks", "water"],
    ["cat", "is", "an", "animal"],
    ["dog", "is", "an", "animal"],
    ["lion", "is", "a", "wild", "animal"],
    ["tiger", "is", "a", "wild", "animal"],
    ["car", "is", "a", "vehicle"],
    ["bus", "is", "a", "vehicle"],
    ["train", "is", "a", "vehicle"],
    ["apple", "is", "a", "fruit"],
    ["mango", "is", "a", "fruit"],
]

start = time.time()

cbow = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=2,
    min_count=1,
    sg=0,
    workers=4,
    epochs=100,
)

cbow_time = time.time() - start

start = time.time()

skipgram = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=2,
    min_count=1,
    sg=1,
    workers=4,
    epochs=100,
)

skipgram_time = time.time() - start


def test_model(model, name):

    print("\n==============================")
    print(name)
    print("==============================")

    print("\nWord Similarity")

    pairs = [("king", "queen"), ("cat", "dog"), ("car", "bus"), ("apple", "mango")]

    for word1, word2 in pairs:
        score = model.wv.similarity(word1, word2)
        print(word1, "<->", word2, ":", round(score, 4))

    print("\nMost Similar Words")

    for word in ["king", "cat", "car"]:
        print("\n", word)

        results = model.wv.most_similar(word, topn=3)

        for result, score in results:
            print(result, ":", round(score, 4))

    print("\nWord Analogy")

    result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)

    print("king - man + woman")

    for word, score in result:
        print(word, ":", round(score, 4))


test_model(cbow, "CBOW")

test_model(skipgram, "SKIP-GRAM")


print("\n==============================")
print("PERFORMANCE COMPARISON")
print("==============================")

print("CBOW Training Time    :", round(cbow_time, 4), "seconds")
print("Skip-Gram Training Time:", round(skipgram_time, 4), "seconds")

print("\nVocabulary Size")
print("CBOW      :", len(cbow.wv))
print("Skip-Gram :", len(skipgram.wv))

print("\nSimilarity Comparison")

pairs = [("king", "queen"), ("cat", "dog"), ("car", "bus"), ("apple", "mango")]

for word1, word2 in pairs:

    cbow_score = cbow.wv.similarity(word1, word2)
    skipgram_score = skipgram.wv.similarity(word1, word2)

    print(word1, "<->", word2)

    print("CBOW      :", round(cbow_score, 4))

    print("Skip-Gram :", round(skipgram_score, 4))
