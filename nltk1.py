from nltk.stem import PorterStemmer, SnowballStemmer

stemmer = PorterStemmer()

words = ["play", "playing", "played", "plays", "studies", "studying"]

for word in words:
    print(word, "|", stemmer.stem(word))


import spacy

nlp = spacy.load("en_core_web_sm")
text = "hi, my name is angad kumar maurya darlings"
doc = nlp(text)
print()
for token in doc:
    print(token.text, "->", token.lemma_)


# stop words
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

print(stop_words)


# remove stop words

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "This is a very simple example of NLP."

tokens = word_tokenize(text)

stop_words = set(stopwords.words("english"))

filtered_words = []

for word in tokens:
    if word.lower() not in stop_words:
        filtered_words.append(word)

print(filtered_words)


# puncutuation handling
import string
from nltk.tokenize import word_tokenize

text = "Hello, world! I love NLP."

tokens = word_tokenize(text)

clean_tokens = []

for token in tokens:
    if token not in string.punctuation:
        clean_tokens.append(token)

print(clean_tokens)


# oov(oout of vocabulary)

vocabulary = {"i", "love", "nlp", "python"}

text = "I love ChatGPT and Python"

tokens = word_tokenize(text.lower())

for token in tokens:
    if token not in vocabulary:
        print(token, "-> OOV")


# handling oov
vocabulary = {"i", "love", "nlp", "python"}

tokens = ["i", "love", "chatgpt", "python"]

processed = []

for token in tokens:
    if token in vocabulary:
        processed.append(token)
    else:
        processed.append("<UNK>")

print(processed)


# normalization
text = "I LOVE NLP"
text = text.lower()
print(text)


# bag of words
from sklearn.feature_extraction.text import CountVectorizer

documents = ["I love NLP", "I love Python"]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())


# n - gram
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

text = "I love natural language processing"

tokens = word_tokenize(text)

bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("Bigrams:", bigrams)
print("Trigrams:", trigrams)


# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["I love NLP", "I love Python", "Python is interesting"]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())


# whole pipeline
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "The students are studying Natural Language Processing!"

# 1. Normalization
text = text.lower()

# 2. Tokenization
tokens = word_tokenize(text)

# 3. Punctuation removal
tokens = [word for word in tokens if word not in string.punctuation]

# 4. Stop-word removal
stop_words = set(stopwords.words("english"))

tokens = [word for word in tokens if word not in stop_words]

# 5. Stemming
stemmer = PorterStemmer()

stemmed = [stemmer.stem(word) for word in tokens]

# 6. Lemmatization
lemmatizer = WordNetLemmatizer()

lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

print("Original:", text)
print("Tokens:", tokens)
print("Stemmed:", stemmed)
print("Lemmatized:", lemmatized)
