import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import string

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
punctuation = set(string.punctuation)


text = "my name is angad kumar maurya ? . ? "

words = word_tokenize(text)

filtered_words = [word for word in words if word.lower() not in stop_words]
filtered_words1 = [word for word in words if word.lower() not in punctuation]

print("oringinal text: ", text)
print("".join(filtered_words))
print("".join(filtered_words1))
