import pandas as pd
import numpy as np
import time
import sys
from NaiveBayes import NaiveBayes


# Function to read data
def load_dataset(path):
    df = pd.read_csv(path, header=0, delimiter=', ')
    return df


def split_data(data, train_ratio=0.7):
    p = data[data['sentimentLabel'] == 0]
    n = data[data['sentimentLabel'] == 1]

    m_p = np.random.rand(len(p)) < train_ratio
    m_n = np.random.rand(len(n)) < train_ratio

    train = pd.concat([p[m_p], n[m_n]])
    test = pd.concat([p[~m_p], n[~m_n]])
    return train, test
