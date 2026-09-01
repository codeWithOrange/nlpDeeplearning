from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("french")

words = ["manger", "manges", "mangeons", "mangeraient"]

for word in words:
    print(word, "→", stemmer.stem(word))

print(stemmer.languages)

# hindi language
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

text = "लड़कियाँ स्कूलों में पढ़ रही हैं"

factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("hi")

text = normalizer.normalize(text)

tokens = indic_tokenize.trivial_tokenize(text)

print(tokens)


# using stanza
import stanza

stanza.download("hi")

nlp = stanza.Pipeline("hi")

doc = nlp("लड़कियाँ स्कूल जा रही हैं।")

for sentence in doc.sentences:
    for word in sentence.words:
        print(word.text, "→", word.lemma)
