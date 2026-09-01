"""
 Load pretrained Word2Vec or GloVe embeddings, select a set of words, compute their embeddings,
and visualize relationships in 2-D space using PCA or t-SNE
"""

import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model = api.load("glove-wiki-gigaword-100")

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

embeddings = [model[word] for word in words]

pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 7))

for i, word in enumerate(words):
    x = vectors_2d[i, 0]
    y = vectors_2d[i, 1]

    plt.scatter(x, y)
    plt.annotate(word, (x, y), xytext=(5, 5), textcoords="offset points")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("GloVe Word Embeddings Visualization using PCA")
plt.grid(True)
plt.show()
