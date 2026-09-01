from contractions import fix
import re
from nltk import word_tokenize

text = "Rs. 1,000 ten email-id E-mail"


words = word_tokenize(text)
print(words)
dictionary = {
    "Rs. 1,000": "1000",
    "ten": "10",
    "email-id": "email id",
    "E-mail": "email",
}

for contraction, expanded in dictionary.items():
    text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
print(text)
