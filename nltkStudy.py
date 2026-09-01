import nltk

"""
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
"""

# word tokenize
from nltk.tokenize import word_tokenize, sent_tokenize

text = "angad kumar maurya"

tokens = word_tokenize(text)
print(tokens)


text = """
i am angad kumar maurya , hello darling
i am angad kumar maurya, hello darling 1 , 2 3
"""

sentences = sent_tokenize(text)
print(sentences)


# stemming
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["playing", "played", "plays", "studies", "studying"]

for word in words:
    print(word, "->", stemmer.stem(word))


# lematizatoin
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ["cats", "dogs", "studies", "playing"]

for word in words:
    print(word, "->", lemmatizer.lemmatize(word))
