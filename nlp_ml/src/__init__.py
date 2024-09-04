import embedding
import DB


def embed(first_sentence, second_sentence):
    full_story: str = "".join([first_sentence, second_sentence])
    embedding.embed_story(full_story)

def load():
    embedding.load_model("word2vec.model")

def main():
    DATABASE: DB.DataBase = DB.DataBase()
    DATABASE.get_Stories(embed)

    load() # load after embedding and saving information to the .model file

if __name__ == "__main__":
    main()
