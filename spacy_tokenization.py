import spacy

nlp = spacy.load("en_core_web_sm")

text = """
hello darling , 
i am angad kumar maurya
hello darling , 
i am angad kumar maurya
"""


doc = nlp(text)

for sent in doc.sents:
    print(sent)
print()
for token in doc:
    print(token.text)


# information of token
for token in doc:
    print(token.text, token.pos_)

# token index
for token in doc:
    print(token.text, token.i)

# character position
for token in doc:
    print(token.text, token.idx)


print(len(doc))
