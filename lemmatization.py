from nltk.stem import WordNetLemmatizer
import nltk

lt = WordNetLemmatizer()

nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
words = [
    "eating",
    "continuous",
    "eats",
    "ate",
    "adjustable",
    "rafting",
    "ability",
    "meeting",
]
for word in words:
    print(word, "->", lemmatizer.lemmatize(word, pos="v"))
