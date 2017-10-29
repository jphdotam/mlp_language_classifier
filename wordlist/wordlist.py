import unidecode
import numpy as np
import string

class WordList():

    def __init__(self,filename,maxlen):
        self.filename = filename
        self.maxlen = maxlen
        self.ascii_wordlist = []
        self.onehot_wordlist = []

        with open(filename,'r',encoding = "ISO-8859-1") as f:
            for _line in f:
                word = f.readline()
                word = unidecode.unidecode(word[:-1]).lower()
                if word.isalpha() and len(word) <= self.maxlen:
                    self.ascii_wordlist.append(word)
                    self.onehot_wordlist.append(self.string_to_onehot(word,self.maxlen))

        self.length = len(self.onehot_wordlist)

    @staticmethod
    def string_to_onehot(word,maxlen):
        onehot = np.array(WordList.string_vectorizer(word)).flatten()
        onehot = np.pad(onehot, (0, (maxlen * 26) - len(onehot)), 'constant')
        return onehot

    @staticmethod
    def string_vectorizer(strng, alphabet=string.ascii_lowercase):
        vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in strng]
        return vector