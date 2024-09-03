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

    def get_Stories(self):
        self.CURSOR.execute("SELECT * FROM two_sentence_horror.stories_INFO;")

        print(self.CURSOR.fetchone())


if __name__ == "__main__":
    DB = DataBase()

