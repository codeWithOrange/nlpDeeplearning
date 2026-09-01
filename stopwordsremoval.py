import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


text = "my name is angad kumar maurya"

words = word_tokenize(text)
filtered_words = [word for word in words if word.lower() not in stop_words]

print("oringinal text: ", text)
print(" ".join(filtered_words))
