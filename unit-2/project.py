import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from gensim.downloader import load

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. CORPUS

sentences = [
    ["king", "is", "a", "man"],
    ["queen", "is", "a", "woman"],
    ["king", "rules", "the", "kingdom"],
    ["queen", "rules", "the", "kingdom"],
    ["man", "is", "strong"],
    ["woman", "is", "strong"],
    ["boy", "is", "young"],
    ["girl", "is", "young"],
    ["cat", "drinks", "milk"],
    ["cat", "eats", "fish"],
    ["cat", "likes", "food"],
    ["dog", "drinks", "milk"],
    ["dog", "eats", "meat"],
    ["dog", "likes", "food"],
    ["lion", "is", "a", "wild", "animal"],
    ["tiger", "is", "a", "wild", "animal"],
    ["car", "is", "a", "vehicle"],
    ["bus", "is", "a", "vehicle"],
    ["train", "is", "a", "vehicle"],
    ["apple", "is", "a", "fruit"],
    ["mango", "is", "a", "fruit"],
]


# 2. CBOW WORD2VEC

cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=2,
    min_count=1,
    sg=0,  # 0 = CBOW
    workers=4,
    epochs=100,
)

print("\n========== CBOW ==========")

print("Embedding of cat:")
print(cbow_model.wv["cat"])


# 3. SKIP-GRAM WORD2VEC

skipgram_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=2,
    min_count=1,
    sg=1,  # 1 = Skip-Gram
    workers=4,
    epochs=100,
)

print("\n========== SKIP-GRAM ==========")

print("Embedding of cat:")
print(skipgram_model.wv["cat"])


# 4. SEMANTIC SIMILARITY

print("\n========== SEMANTIC SIMILARITY ==========")

similarity = skipgram_model.wv.similarity("cat", "dog")

print("Similarity(cat, dog):", similarity)


similarity = skipgram_model.wv.similarity("king", "queen")

print("Similarity(king, queen):", similarity)


# 5. MOST SIMILAR WORDS

print("\n========== MOST SIMILAR WORDS ==========")

print("Words similar to cat:")

for word, score in skipgram_model.wv.most_similar("cat", topn=5):
    print(word, ":", score)


# 6. WORD ANALOGY

print("\n========== WORD ANALOGY ==========")

result = skipgram_model.wv.most_similar(
    positive=["king", "woman"], negative=["man"], topn=5
)

print("king - man + woman ≈ ?")

for word, score in result:
    print(word, ":", score)


# 7. VECTOR SPACE MODEL

print("\n========== VECTOR SPACE ==========")

words = [
    "king",
    "queen",
    "man",
    "woman",
    "cat",
    "dog",
    "lion",
    "tiger",
    "car",
    "bus",
    "train",
    "apple",
    "mango",
]

vectors = np.array([skipgram_model.wv[word] for word in words])

print("Number of words:", len(words))
print("Embedding dimensions:", vectors.shape[1])

print("Vector matrix shape:", vectors.shape)


# 8. PCA

print("\n========== PCA ==========")

pca = PCA(n_components=2)

vectors_2d = pca.fit_transform(vectors)

print("Original shape:", vectors.shape)
print("After PCA:", vectors_2d.shape)


plt.figure(figsize=(10, 7))

for i, word in enumerate(words):

    x = vectors_2d[i, 0]
    y = vectors_2d[i, 1]

    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word)

plt.title("Word Embeddings using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()


# 9. t-SNE

print("\n========== t-SNE ==========")

tsne = TSNE(n_components=2, perplexity=5, random_state=42)

vectors_tsne = tsne.fit_transform(vectors)


plt.figure(figsize=(10, 7))

for i, word in enumerate(words):

    x = vectors_tsne[i, 0]
    y = vectors_tsne[i, 1]

    plt.scatter(x, y)
    plt.text(x + 0.5, y + 0.5, word)

plt.title("Word Embeddings using t-SNE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.show()


# 10. GLOVE

print("\n========== GLOVE ==========")

print("Loading pretrained GloVe model...")

glove_model = load("glove-wiki-gigaword-50")

print("GloVe vector for 'king':")

print(glove_model["king"])


# 11. GLOVE SIMILARITY

print("\nGloVe Similarity")

print("king vs queen:", glove_model.similarity("king", "queen"))

print("cat vs dog:", glove_model.similarity("cat", "dog"))


# 12. GLOVE MOST SIMILAR

print("\nGloVe words similar to cat:")

for word, score in glove_model.most_similar("cat", topn=5):

    print(word, ":", score)


# 13. GLOVE ANALOGY

print("\nGloVe Analogy")

result = glove_model.most_similar(positive=["king", "woman"], negative=["man"], topn=5)

for word, score in result:

    print(word, ":", score)


print("\n========== PROGRAM FINISHED ==========")
