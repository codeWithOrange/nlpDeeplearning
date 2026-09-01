from operator import contains
from nltk.tokenize import word_tokenize

text = "Angad Angad  kumar maurya"
# text = "".join(text.split())
words = word_tokenize(text)
print(text.lower())
print(words)


freq = {}
for word in words:
    freq[word] = freq.get(word, 0) + 1


print(freq)


import re
import contractions

text = "I'm learning NLP, I don't know why you're studying it. I can't understand it"
contractions1 = {
    "I'm": "I am",
    "don't": "do not",
    "can't": "can not",
    "you're": "you are",
    "won't": "will not",
    "doesn't": "does not",
}
ob = contractions.fix(text)
print(ob)
"""
for contraction, expanded in contractions1.items():
    text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
print(text)
"""
