from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords


class NLP:
    def __init__(self) -> None:
        self.TEST_STORY: str = """When the tsunamis hit, most people were evacuated. Feeling the trickle of water rise to my chin, I let go of the prison bars in bitter resentment."""
        
        self.tokenize(self.TEST_STORY)

    def tokenize(self, story: str) -> None:
        self.SENTENCE_TOKENS: list[str] = sent_tokenize(story)

        for sentence in self.SENTENCE_TOKENS:
            WORD_TOKENS: list[str] = word_tokenize(sentence)
            
            self.WITHOUT_STOPWORDS: list[str] = self.filter_Stop_Words(WORD_TOKENS)
            print(self.Stem_Words(self.WITHOUT_STOPWORDS))
        
    def filter_Stop_Words(self, Word_tokens: list[str]) -> list[str]:
        self.STOP_WORDS: set = set(stopwords.words("english"))
        
        self.result: list[str] = [word for word in Word_tokens if word.casefold() not in self.STOP_WORDS]

        return self.result
    
    def Stem_Words(self, words: list[str]) -> list:
        self.ENGLISH_STEMMER: EnglishStemmer = EnglishStemmer()

        self.STEMMED_WORDS: list = [self.ENGLISH_STEMMER.stem(word) for word in words]

        return self.STEMMED_WORDS


if __name__ == "__main__":
    Natural_Language_Processor: NLP = NLP()
