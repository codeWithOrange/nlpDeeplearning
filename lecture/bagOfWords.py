from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning",
    "I love NLP",
    "machine learning interesting",
]

vectorizer = CountVectorizer()


x = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(x.toarray())
