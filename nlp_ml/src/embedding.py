from gensim.models import Word2Vec
import gensim
import nlp


test_story: str = """
Word2Vec is a popular technique for natural language processing. It creates word embeddings for text."""

Natural_Language_Processor: nlp.NLP = nlp.NLP(test_story)
processed = Natural_Language_Processor.tokenize()

print(processed)
model = Word2Vec(sentences= processed, window=555, min_count=1)

print(model.wv.most_similar("word", topn=555))

