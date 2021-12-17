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

    # Generation of the dictionaries for positive and negative words
    def fit(self, train):
        # 0: negative, 1: positive
        p = train[train['sentimentLabel'] == 1]['tweetText'].to_numpy()
        n = train[train['sentimentLabel'] == 0]['tweetText'].to_numpy()

        # Laplace Smoothing
        self.p_prob = (len(p) + self.laplace) / (len(p) + len(n) + (self.laplace*2))
        self.n_prob = (len(n) + self.laplace) / (len(p) + len(n) + (self.laplace*2))

        # Positive dictionary
        for i in p:
            for word in i.split(" "):
                if word not in self.p_dict:
                    self.p_dict[word] = 1
                else:
                    self.p_dict[word] += 1

        # Negative dictionary
        for i in n:
            for word in i.split(" "):
                if word not in self.n_dict:
                    self.n_dict[word] = 1
                else:
                    self.n_dict[word] += 1

        if self.stop:
            self.stop_words()

        if self.n_words > 0:
            self.p_dict = dict(sorted(self.p_dict.items(), key=itemgetter(1), reverse=True)[:self.n_words])
            self.n_dict = dict(sorted(self.n_dict.items(), key=itemgetter(1), reverse=True)[:self.n_words])
        self.p_sum = sum(self.p_dict.values())
        self.n_sum = sum(self.n_dict.values())
        self.total_dict_words = len(set(self.p_dict.keys()) | set(self.n_dict.keys()))

    def predict(self, test):
        twtstest = test['tweetText'].to_numpy()
        pred = np.zeros(len(twtstest), dtype=int)
        if self.stop:
            for i in range(0, len(twtstest)):
                prob_pos = 1
                prob_neg = 1
                for paraula in twtstest[i].split(" "):
                    if paraula not in words:
                        if paraula in self.p_dict:
                            prob_pos *= ((self.p_dict[paraula] + self.laplace) / (
                                        self.p_sum + (self.total_dict_words * self.laplace)))
                        # We exclude words with a length less than 3
                        if paraula not in self.p_dict and len(paraula) > 2:
                            prob_pos *= ((0 + self.laplace) / (self.p_sum + (self.total_dict_words * self.laplace)))

                        if paraula in self.n_dict:
                            prob_neg *= ((self.n_dict[paraula] + self.laplace) / (
                                        self.n_sum + (self.total_dict_words * self.laplace)))
                        # We exclude words with a length less than 3
                        if paraula not in self.n_dict and len(paraula) > 2:
                            prob_neg *= ((0 + self.laplace) / (self.n_sum + (self.total_dict_words * self.laplace)))
                prob_pos *= self.p_prob
                prob_neg *= self.n_prob
                pred[i] = int(prob_pos > prob_neg)
        else:
            for i in range(0, len(twtstest)):
                prob_pos = 1
                prob_neg = 1
                for paraula in twtstest[i].split(" "):
                    if paraula not in words:
                        if paraula in self.p_dict:
                            prob_pos *= ((self.p_dict[paraula] + self.laplace) / (
                                        self.p_sum + (self.total_dict_words * self.laplace)))
                        # We exclude words with a length less than 3
                        if paraula not in self.p_dict and len(paraula) > 2:
                            prob_pos *= ((0 + self.laplace) / (self.p_sum + (self.total_dict_words * self.laplace)))
                        if paraula in self.n_dict:
                            prob_neg *= ((self.n_dict[paraula] + self.laplace) / (
                                        self.n_sum + (self.total_dict_words * self.laplace)))
                        # We exclude words with a length less than 3
                        if paraula not in self.n_dict and len(paraula) > 2:
                            prob_neg *= ((0 + self.laplace) / (self.n_sum + (self.total_dict_words * self.laplace)))
                prob_pos *= self.p_prob
                prob_neg *= self.n_prob
                pred[i] = int(prob_pos > prob_neg)
        return pred