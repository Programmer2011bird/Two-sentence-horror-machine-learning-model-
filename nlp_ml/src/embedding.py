from gensim.models import Word2Vec
import gensim
import nlp

global model

def embed_story(story: list[str]):
    try:
        model = Word2Vec.load("two_sentence_horror.model")
        print("Loaded existing model")

    except FileNotFoundError:
        print("No existing model found, initializing a new one")
        model = Word2Vec(vector_size=100, window=10, min_count=1, workers=4)

    for _, sentences in enumerate(story):
        Natural_Language_Processor: nlp.NLP = nlp.NLP(sentences)
        processed = Natural_Language_Processor.tokenize()
        
        print(processed)

        model.build_vocab(processed, update=True)
        
        model.train(processed, total_examples=1, epochs=model.epochs)

    model.save("two_sentence_horror.model")
    print("saved")

def load_model(file) -> Word2Vec:
    model = Word2Vec.load(file)
    
    return model
