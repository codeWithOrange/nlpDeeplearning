from gensim.models import Word2Vec
from sklearn.manifold import TSNE

print(dir(Word2Vec))
sentences = [
    ["the", "cat", "drinks", "milk"],
    ["the", "dog", "drinks", "water"],
    ["the", "cat", "eats", "fish"],
    ["the", "dog", "eats", "meat"],
    ["king", "is", "a", "man"],
    ["queen", "is", "a", "woman"],
]
model1 = TSNE()
print(model1)

model = Word2Vec(
    sentences,
    vector_size=40,
    window=2,
    min_count=1,
    sg=1,
)
vector = model.wv["cat"]
print(vector, len(vector))

print(model.wv.most_similar("cat"))
similarity = model.wv.similarity("cat", "dog")
print(similarity)


result = model.wv.most_similar(positive=["king", "woman"], negative=["queen"], topn=1)

print(result)


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = ["king", "queen", "woman", "cat", "dog"]
vectors = [model.wv[word] for word in words]
print(vectors)
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)
print(result)


plt.figure(figsize=(8, 6))

for i, word in enumerate(words):
    x = result[i, 0]
    y = result[i, 1]

    plt.scatter(x, y)
    plt.text(x, y, word)

plt.show()
