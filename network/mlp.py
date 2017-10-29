import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier

from gui.mlp_gui import MLP_gui
from wordlist.wordlist import WordList

class Language_MLP():

    def load_words(self):

        print("Loading wordlists")
        wordlist1 = WordList(filename='./wordlists/{}.txt'.format(self.language1), maxlen=15)
        wordlist2 = WordList(filename='./wordlists/{}.txt'.format(self.langyage2), maxlen=15)

        language1_words = np.stack(wordlist1.onehot_wordlist)
        wordlist2_words = np.stack(wordlist2.onehot_wordlist)

        if self.balance_wordlists:
            min_wordlist_size = min(language1_words.shape[0], wordlist2_words.shape[0])

            np.random.shuffle(language1_words)
            np.random.shuffle(wordlist2_words)

            print("Crooping to {}".format(min_wordlist_size))
            language1_words = language1_words[:min_wordlist_size, ]
            wordlist2_words = wordlist2_words[:min_wordlist_size, ]

        X = np.vstack((language1_words, wordlist2_words))
        y = np.concatenate([np.repeat(0, language1_words.shape[0]), np.repeat(1, wordlist2_words.shape[0])])

        print("{} {} words".format(self.language1.upper(), len(language1_words)))
        print("{} {} words".format(self.langyage2.upper(), len(wordlist2_words)))

        return X,y

    def __init__(self,language1,language2,load_wordlists,load_network,save_network,balance_wordlists=True,hidden_layers=(20,20)):
        self.language1 = language1
        self.langyage2 = language2

        self.load_wordlists = load_wordlists
        self.load_network = load_network
        self.save_network = save_network
        self.balance_wordlists = balance_wordlists
        self.hidden_layers = hidden_layers

        self.pickle_file = "./models/mlp-{}-{}-{}.model".format(language1, language2, hidden_layers)

        if load_wordlists:
            self.X, self.y = self.load_words()

        if load_network:
            print("Loading network from pickle ({})".format(self.pickle_file))
            with open(self.pickle_file, 'rb') as f:
                self.clf = pickle.load(f)
        else:
            print("Training from sratch")
            self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=hidden_layers,random_state=1,verbose=True)
            self.clf.fit(self.X,self.y)

            if save_network:
                print("Saving as pickle ({})".format(self.pickle_file))
                with open(self.pickle_file, 'wb') as f:
                    pickle.dump(self.clf, f)

        if load_wordlists: print("Accuracy: %f" % self.clf.score(self.X, self.y))

        self.gui = MLP_gui(self.clf, language1, language2)