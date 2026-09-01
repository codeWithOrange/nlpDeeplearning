"""Given a set of text documents, implement a preprocessing pipeline that performs tokenization,
lowercasing, noise removal, stopword removal, stemming/lemmatization, and converts the processed
text into numerical representations using Bag-of-Words and TF-IDF"""

import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


documents = [
    "I love Natural Language Processing!",
    "NLP is an interesting field of Artificial Intelligence.",
    "I am learning NLP using Python.",
    "Python is very useful for Natural Language Processing.",
]


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


processed_documents = []
print("\n========== PREPROCESSING ==========\n")
for i, document in enumerate(documents):
    tokens = preprocess(document)
    processed_documents.append(" ".join(tokens))
    print("Document", i + 1)
    print("Original :", document)
    print("Processed:", tokens)
    print()


print("\n========== BAG OF WORDS ==========\n")
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(processed_documents)
print("Vocabulary:")
print(bow_vectorizer.get_feature_names_out())
print("\nBoW Matrix:")
print(bow_matrix.toarray())


print("\n========== TF-IDF ==========\n")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_documents)
print("Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("\n========== TF-IDF WITH WORDS ==========\n")
words = tfidf_vectorizer.get_feature_names_out()
matrix = tfidf_matrix.toarray()
for i, document in enumerate(matrix):
    print("Document", i + 1)
    for j, value in enumerate(document):
        if value > 0:
            print(words[j], ":", round(value, 3))
    print()
