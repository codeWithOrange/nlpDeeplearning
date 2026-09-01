# word2vec

1. Deep learning model that train on huge dataset a technique in NLP that converts words into numbers (vectors) in such a way that:
2. vectors capture information about the meaning of the word based on surrounding words
3. words with similar meaning get similar vectors
4. relationship between words are preserved
5. it learns word meaning from context (surroundign words)
6. words that appear in similar contexts have similar meanings.

# cbow characteristics

1. uses context words to predict center word
2. fast
3. works well with large datasets
4. less effective for rare words
5. used when speed is important

# skip - gram characteristics

1. uses center word to predict context
2. slower than cbow
3. very good at learning rare words
4. produces high-quality embedings
5. preffered when accuracy is more important

# summary

1. cbow - predict the center word
2. skip-gram - predict surrounding words
3. embedding size - length of numeric vector
4. window size = k -> context size = 2k
