from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from colorama import Fore, init
import nltk


init(convert= True)
nltk.data.path = ["../Lib"]

class NLP:
    def __init__(self, story: str) -> None:
        self.STORY: str = story
        self.STORY = self.normalize(self.STORY)
        
        print(Fore.LIGHTGREEN_EX + "ORIGINAL : " + Fore.LIGHTCYAN_EX + self.STORY)

    def normalize(self, story: str) -> str:
        return story.lower().replace(',', '').replace(".", "")

    def tokenize(self) -> list[str]:
        SENT_TOKENS = sent_tokenize(self.STORY)
        self.TOKENS = []

        for sentence in SENT_TOKENS:
            WORD_TOKENS: list[str] = word_tokenize(sentence)
            
            self.WITHOUT_STOPWORDS: list[str] = self.filter_Stop_Words(WORD_TOKENS)
            self.POS_TAGS: list[tuple[str, str]] = self.Tag_part_of_speech(self.WITHOUT_STOPWORDS)
            self.LEMMATIZED_WORDS: list[str] = self.Lemmatize_Words(self.POS_TAGS)
            self.TOKENS.append(self.LEMMATIZED_WORDS)

        return self.TOKENS
        
    def filter_Stop_Words(self, Word_tokens: list[str]) -> list[str]:
        self.STOP_WORDS: set = set(stopwords.words("english"))
        
        self.result: list[str] = [word for word in Word_tokens if word.casefold() not in self.STOP_WORDS]

        return self.result
    
    def wordnet_pos(self, TAG: str) -> str:
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

    def Lemmatize_Words(self, POS: list[tuple[str, str]]) -> list[str]:
        self.LEMMATIZED_WORDS: list[str] = []

        for _, (self.WORD, self.POS) in enumerate(POS):
            self.ENGLISH_LEMMATIZER: WordNetLemmatizer = WordNetLemmatizer()
            
            self.WORDNET_POS: str = self.wordnet_pos(self.POS)
            self.LEMMATIZED_WORDS.append(self.ENGLISH_LEMMATIZER.lemmatize(self.WORD, self.WORDNET_POS))

        return self.LEMMATIZED_WORDS

    def Tag_part_of_speech(self, wordTokens: list[str]) -> list[tuple[str, str]]:
        self.POS_TAGS: list[tuple[str, str]] = pos_tag(wordTokens)

        return self.POS_TAGS
