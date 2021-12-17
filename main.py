import pandas as pd
import numpy as np
import time
import sys
from NaiveBayes import NaiveBayes


# Function to read data
def load_dataset(path):
    df = pd.read_csv(path, header=0, delimiter=';')
    return df


def split_data(data, train_ratio=0.7):
    p = data[data['sentimentLabel'] == 0]
    n = data[data['sentimentLabel'] == 1]

    m_p = np.random.rand(len(p)) < train_ratio
    m_n = np.random.rand(len(n)) < train_ratio

    train = pd.concat([p[m_p], n[m_n]])
    test = pd.concat([p[~m_p], n[~m_n]])
    return train, test


# To calculate the "confusion matrix"
def metrics(y_true, pred):
    conf_matrix = np.array([0.0, 0.0, 0.0, 0.0])  # true_positive, false_positive, false_negative, true_negative
    if len(y_true) == len(pred):
        y_true_neg = np.where((y_true == 0) | (y_true == 1), y_true ^ 1, y_true)
        pred_n = np.where((pred == 0) | (pred == 1), pred ^ 1, pred)
        c0 = np.count_nonzero(np.logical_and(pred, y_true))
        c1 = np.count_nonzero(np.logical_and(pred, y_true_neg))
        c2 = np.count_nonzero(np.logical_and(pred_n, y_true))
        c3 = np.count_nonzero(np.logical_and(pred_n, y_true_neg))
        conf_matrix[0] += c0 / (c0 + c1)
        conf_matrix[1] += c1 / (c0 + c1)
        conf_matrix[2] += c2 / (c2 + c3)
        conf_matrix[3] += c3 / (c2 + c3)
        return conf_matrix, (np.count_nonzero(y_true == pred)/len(pred))
    return conf_matrix, 0


# K-fold Cross Validation
def cross_validation(data, k=5, words=2500, stop=False):
    mean_accuracy = 0.0
    mean_confidence = np.array([0.0, 0.0, 0.0, 0.0])
    shuffled = data.sample(frac=1)
    partitions = np.array_split(shuffled, k)

    for i in range(0, k):
        train = []
        for j in range(0, k):
            if j != i:
                train.append(partitions[j])
        train = pd.concat(train)
        test = partitions[i]
        nb = NaiveBayes(words, stop=stop)
        nb.fit(train)
        prediction = nb.predict(test)
        conf_matrix, acc = metrics(test['sentimentLabel'].to_numpy(), prediction)
        mean_accuracy += acc
        mean_confidence += conf_matrix
    return (mean_confidence / k), (mean_accuracy / k)


def main():
    csv = 'FinalStemmedSentimentAnalysisDataset.csv'
    print("Reading file:" + csv)
    tweets = load_dataset(csv)
    print("\n===== Data =====")
    print("- Total number of samples: " + str(tweets.shape[0]))
    print("- Number of attributes: " + str(tweets.shape[1]))
    print("\n- Percentage of positive tweets: " + str(round((tweets[tweets['sentimentLabel'] == 1].shape[0] / tweets.shape[0]) * 100, 3)))
    print("- Percentage of negative tweets: " + str(round((tweets[tweets['sentimentLabel'] == 0].shape[0] / tweets.shape[0]) * 100, 3)))
    print("\n===== Cleaning NaN values =====")
    tweets = tweets.dropna()
    print("- Total number of samples: " + str(tweets.shape[0]))
    print("- Number of attributes " + str(tweets.shape[1]))
    print("\n- Percentage of positive tweets: " + str(round((tweets[tweets['sentimentLabel'] == 1].shape[0] / tweets.shape[0]) * 100, 3)))
    print("- Percentage of negative tweets: " + str(round((tweets[tweets['sentimentLabel'] == 0].shape[0] / tweets.shape[0]) * 100, 3)))

    print("\n===== Models =====")
    print("- Naive bayes without removing stopwords:")
    train, test = split_data(tweets)
    print("\tTraining model  ...", end="")
    start = time.time()
    nb = NaiveBayes(2500)
    nb.fit(train)
    end = time.time()
    temps1 = round(end - start, 4)
    print("\t" + str(temps1) + " seconds")
    print("\tModel prediction ...", end="")
    start = time.time()
    pred = nb.predict(test)
    end = time.time()
    temps2 = round(end - start, 4)
    print("\t" + str(temps2) + " seconds")
    print("\t-----------------------------------------------")
    print("\tTotal time: " + str(temps1 + temps2) + " seconds")
    print("\t-----------------------------------------------")
    confm, acc = metrics(test['sentimentLabel'].to_numpy(), pred)
    print("\tAccuracy: " + str(round(acc * 100, 3)))
    print("\tTruePositives: " + str(round(confm[0] * 100, 3)) + "\tFalsePositives: " + str(round(confm[1] * 100, 3)))
    print("\tFalseNegatives: " + str(round(confm[2] * 100, 3)) + "\tTrueNegatives: " + str(round(confm[3] * 100, 3)))

    print("\n- Naive bayes removing stopwords:")
    print("\tTraining model  ...", end="")
    start = time.time()
    nb = NaiveBayes(2500, stop=True)
    nb.fit(train)
    end = time.time()
    temps1 = round(end - start, 4)
    print("\t" + str(temps1) + " seconds")
    print("\tModel prediction ...", end="")
    start = time.time()
    pred = nb.predict(test)
    end = time.time()
    temps2 = round(end - start, 4)
    print("\t" + str(temps2) + " seconds")
    print("\t-----------------------------------------------")
    print("\tTotal time: " + str(temps1 + temps2) + " seconds")
    print("\t-----------------------------------------------")
    confm, acc = metrics(test['sentimentLabel'].to_numpy(), pred)
    print("\tAccuracy: " + str(round(acc * 100, 3)))
    print("\tTruePositives: " + str(round(confm[0] * 100, 3)) + "\tFalsePositives: " + str(round(confm[1] * 100, 3)))
    print("\tFalseNegatives: " + str(round(confm[2] * 100, 3)) + "\tTrueNegatives: " + str(round(confm[3] * 100, 3)))

    print("\n- Effect of train-test set size:")
    part = [0.5, 0.6, 0.7, 0.8]
    for i in range(0, len(part)):
        print("\n\t" + str(part[i] * 100) + "-" + str(100 - (part[i] * 100)) + ":")
        train, test = split_data(tweets, train_ratio=part[i])
        print("\tTraining model  ...", end="")
        start = time.time()
        nb = NaiveBayes(2500, stop=True)
        nb.fit(train)
        end = time.time()
        temps1 = round(end - start, 4)
        print("\t" + str(temps1) + " seconds")
        print("\tModel prediction ...", end="")
        start = time.time()
        pred = nb.predict(test)
        end = time.time()
        temps2 = round(end - start, 4)
        print("\t" + str(temps2) + " seconds")
        print("\t-----------------------------------------------")
        print("\tTotal time: " + str(temps1 + temps2) + " seconds")
        print("\t-----------------------------------------------")
        confm, acc = metrics(test['sentimentLabel'].to_numpy(), pred)
        print("\tAccuracy: " + str(round(acc * 100, 3)))
        print("\tTruePositives: " + str(round(confm[0] * 100, 3)) + "\tFalsePositives: " + str(round(confm[1] * 100, 3)))
        print("\tFalseNegatives: " + str(round(confm[2] * 100, 3)) + "\tTrueNegatives: " + str(round(confm[3] * 100, 3)))

    print("\n- Changing the size of dictionaries with K-fold K = 5.")
    paraules = [500, 2000, 3500, 5000, 6500, 8000, 0]
    for i in range(0, len(paraules)):
        if paraules[i] > 0:
            print("\n\t" + str(paraules[i]) + " words:")
        else:
            print("\n\tAll words:")
        print("\tCalculating ...", end="")
        start = time.time()
        confm, acc = cross_validation(tweets, 5, words=paraules[i], stop=True)
        end = time.time()
        temps2 = round(end - start, 4)
        print("\t" + str(temps2) + " seconds")
        print("\tAccuracy: " + str(round(acc * 100, 3)))
        print("\tTruePositives: " + str(round(confm[0] * 100, 3)) + "\tFalsePositives: " + str(round(confm[1] * 100, 3)))
        print("\tFalseNegatives: " + str(round(confm[2] * 100, 3)) + "\tTrueNegatives: " + str(round(confm[3] * 100, 3)))


# If you want to re-write the .txt file, set "p = True". It takes 15 min to finish.
p = True
if p:
    original_stdout = sys.stdout
    # Creates a txt file to store the result, don't show anything by console
    with open('NaiveBayesOutputCorrect.txt', 'w') as f:
        sys.stdout = f
        main()
        sys.stdout = original_stdout
    print("Check NaiveBayesOutput.txt")
else:
    main()
