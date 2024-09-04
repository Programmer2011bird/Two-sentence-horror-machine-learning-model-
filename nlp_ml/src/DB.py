from gensim.models import Word2Vec
from config import db_config
from colorama import Fore
import mysql.connector
import gensim


class DataBase:
    def __init__(self) -> None:
        print(db_config)
        self.DB = mysql.connector.connect(**db_config)
        
        self.DB.database = "two_sentence_horror"

        self.CURSOR = self.DB.cursor()
        
        print(Fore.GREEN + "DATABASE CONNECTED SUCCESSFULLY")

        self.get_Stories()

    def get_Stories(self, func= None) -> None:
        self.CURSOR.execute("SELECT first_sentence, second_sentence FROM two_sentence_horror.stories_INFO;")
        
        for story_info in self.CURSOR.fetchall():
            self.FIRST_SENTENCE: str = str(story_info[0])
            self.SECOND_SENTENCE: str = str(story_info[1])

            if func != None:
                func(self.FIRST_SENTENCE, self.SECOND_SENTENCE)

