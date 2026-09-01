from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Natural language processing is a branch of artificial intelligence.",
    "Machine laerning is used for text classification.",
    "Deep learning can improve natural language",
]


vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(corpus)


words = vectorizer.get_feature_names_out()


print("words:")
print(words)

print("Tf idf matrix: ")
print(matrix.toarray())
