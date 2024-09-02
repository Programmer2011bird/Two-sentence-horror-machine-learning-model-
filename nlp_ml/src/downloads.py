import nltk

def main():
    nltk.download("punkt_tab", "../Lib")
    nltk.download("stopwords", "../Lib")
    nltk.download('averaged_perceptron_tagger_eng', "../Lib")
    nltk.download('wordnet', "../Lib")

if __name__ == "__main__":
    main()
