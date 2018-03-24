from cleanData import clean_str
import io
import numpy as np
from CBOWloader import word2idx

EMEDDING_DIMENSION = 100

def prepare_data(filename):
    word_emb_mat = np.loadtxt("w_emb_mat.txt")
    X = []
    y_pred = []
    vocab_size = len(word2idx)
    print(vocab_size)
    with io.open(filename, "r", encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            ID, label, sentence = line.split('\t')
            label_idx = 1 if label == 'pos' else 0  # 1 for pos and 0 for neg

            sentence = clean_str(sentence)

            x_temp = np.zeros((EMEDDING_DIMENSION, ), dtype=np.int)
            for word in sentence.split():
                if word in word2idx:
                    x_temp = x_temp + word_emb_mat[word2idx[word]]

            X.append(x_temp)
            y_pred.append(label_idx)

    return X, y_pred


prepare_data("twitter-sentiment.csv")