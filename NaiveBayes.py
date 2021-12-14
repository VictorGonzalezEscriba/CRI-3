import numpy as np
from nltk.corpus import stopwords
from operator import itemgetter
words = set(stopwords.words('english'))


class NaiveBayes:
    def __init__(self, n_words, stop=False):
        # Probability of positive message
        self.p_prob = 0
        # Probability of negative message
        self.n_prob = 0
        # Sum of all words and their occurrences in positive tweets
        self.p_sum = 0
        # Sum of all words and their occurrences in negative tweets
        self.n_sum = 0
        # Total words at the dictionaries
        self.total_dict_words = 0
        # Number of possible words at dictionary
        self.n_words = n_words
        # Dictionary with positive words
        self.p_dict = {}
        # Dictionary with negative words
        self.n_dict = {}
        # Value of Laplace Smoothing
        self.laplace = 1
        # Use stopwords or not
        self.stop = stop

    def stop_words(self):
        for word in words:
            if word in self.p_dict:
                del self.p_dict[word]
            if word in self.n_dict:
                del self.n_dict[word]

    # Generation of the dictionaries
    def fit(self, train):
        p = train[train['sentimentLabel'] == 1]['tweetText'].to_numpy()
        n = train[train['sentimentLabel'] == 0]['tweetText'].to_numpy()
        self.p_prob = (len(p) + self.laplace) / (len(p) + len(n) + (self.laplace*2))
        self.n_prob = (len(n) + self.laplace) / (len(p) + len(n) + (self.laplace*2))

        # Positive dictionary
        for i in p:
            for paraula in i.split(" "):
                if paraula not in self.p_dict:
                    self.p_dict[paraula] = 1
                else:
                    self.p_dict[paraula] += 1

        # Negative dictionary
        for i in n:
            for paraula in i.split(" "):
                if paraula not in self.n_dict:
                    self.n_dict[paraula] = 1
                else:
                    self.n_dict[paraula] += 1

        if self.stop:
            self.stop_words()

        if self.n_words > 0:
            self.p_dict = dict(sorted(self.p_dict.items(), key=itemgetter(1), reverse=True)[:self.n_words])
            self.n_dict = dict(sorted(self.n_dict.items(), key=itemgetter(1), reverse=True)[:self.n_words])
        self.p_sum = sum(self.p_dict.values())
        self.n_sum = sum(self.n_dict.values())
        self.total_dict_words = len(set(self.p_dict.keys()) | set(self.n_dict.keys()))