from gensim.models import Word2Vec
import gensim
import nlp


def embed_story(story: str):
    Natural_Language_Processor: nlp.NLP = nlp.NLP(story)
    processed = Natural_Language_Processor.tokenize()

    print(processed)
    model = Word2Vec(sentences= processed, window=5, min_count=0)
    
    for index in range(len(model.wv)):
        print(model.wv[index])

    model.save("word2vec.model")

def load_model(file):
    model = Word2Vec.load(file)

    for index in range(len(model.wv)):
        print(model.wv[index])

