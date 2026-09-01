import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = [
    "painting",
    "walking",
    "dressing",
    "likely",
    "children",
    "whom",
    "good",
    "ate",
    "fishing",
]

for word in words:
    print(word, "->", stemmer.stem(word))


for word in words:
    print(word, "->", lemmatizer.lemmatize(word, pos="v"))


text = "latha is very multi talented girl she is good skills like danicing , running, singing playingm. she also likes eating pav bhagi . she has a habit of fishing and swimming too . besiced all this she is a wonderful at cooking too."


words = word_tokenize(text)
print(words)

for word in words:
    print(word, stemmer.stem(word))

for word in words:
    print(word, "->", lemmatizer.lemmatize(word, pos="v"))
