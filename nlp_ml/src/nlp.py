from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk

nltk.data.path = ["../Lib"]

class NLP:
    def __init__(self) -> None:
        self.TEST_STORY: str = """When the tsunamis hit, most people were evacuated. Feeling the trickle of water rise to my chin, I let go of the prison bars in bitter resentment."""
        
        self.tokenize(self.TEST_STORY)

    def tokenize(self, story: str) -> None:
        self.SENTENCE_TOKENS: list[str] = sent_tokenize(story)

        for sentence in self.SENTENCE_TOKENS:
            WORD_TOKENS: list[str] = word_tokenize(sentence)
            
            self.WITHOUT_STOPWORDS: list[str] = self.filter_Stop_Words(WORD_TOKENS)
            self.POS_TAGS: list[tuple[str, str]] = self.Tag_part_of_speech(self.WITHOUT_STOPWORDS)
            self.LEMMATIZED_WORDS = self.Lemmatize_Words(self.POS_TAGS)

            print(self.LEMMATIZED_WORDS)
        
    def filter_Stop_Words(self, Word_tokens: list[str]) -> list[str]:
        self.STOP_WORDS: set = set(stopwords.words("english"))
        
        self.result: list[str] = [word for word in Word_tokens if word.casefold() not in self.STOP_WORDS]

        return self.result
    
    def wordnet_pos(self, TAG: str):
        if TAG.startswith('J'):
            return wordnet.ADJ
        
        elif TAG.startswith('V'):
            return wordnet.VERB
        
        elif TAG.startswith('N'):
            return wordnet.NOUN
        
        elif TAG.startswith('R'):
            return wordnet.ADV
        
        else:
            return wordnet.NOUN

    def Lemmatize_Words(self, POS: list[tuple[str, str]]) -> list:
        self.LEMMATIZED_WORDS: list = []

        for _, (self.WORD, self.POS) in enumerate(POS):
            self.ENGLISH_LEMMATIZER: WordNetLemmatizer = WordNetLemmatizer()
            
            self.WORDNET_POS = self.wordnet_pos(self.POS)
            self.LEMMATIZED_WORDS.append(self.ENGLISH_LEMMATIZER.lemmatize(self.WORD, self.WORDNET_POS))

        return self.LEMMATIZED_WORDS

    def Tag_part_of_speech(self, wordTokens: list[str]) -> list[tuple[str, str]]:
        self.POS_TAGS: list[tuple[str, str]] = pos_tag(wordTokens)

        return self.POS_TAGS


if __name__ == "__main__":
    Natural_Language_Processor: NLP = NLP()
