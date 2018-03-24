from cleanData import clean_str
import io
from torch.utils.data import Dataset, DataLoader
import torch

bsz = 32
CONTEXT = 2

word2idx = {'padding': 0}
idx2word = {0: 'padding'}


class TwitterData(Dataset):

    def __init__(self, filename):
        X = []
        y_pred = []
        index = 1

        with io.open(filename, "r", encoding="ISO-8859-1") as f:
            next(f)
            for line in f:
                _, _, sentence = line.split('\t')
                sentence = clean_str(sentence)
                wordsList = sentence.split()
                index = self._addToDictionaries(index, wordsList)
                self._extractContext(X, y_pred, wordsList)

        self.x_data = torch.LongTensor(X)
        self.y_data = torch.LongTensor(y_pred)
        self.len = len(self.x_data)

    def _extractContext(self, X, y_pred, wordsList):
        for i in range(len(wordsList)):
            temp_X = []

            if i - CONTEXT < 0:
                pad_count = CONTEXT - i
                for j in range(pad_count):
                    temp_X.append(word2idx['padding'])
                for j in range(CONTEXT - pad_count, 0, -1):
                    temp_X.append(word2idx[wordsList[i - j]])
            else:
                for j in range(CONTEXT, 0, -1):
                    temp_X.append(word2idx[wordsList[i - j]])


            if i + CONTEXT > len(wordsList) - 1:
                pad_count = i + CONTEXT - len(wordsList) + 1
                for j in range(1, CONTEXT - pad_count + 1):
                    temp_X.append(word2idx[wordsList[i + j]])
                for j in range(pad_count):
                    temp_X.append(word2idx['padding'])
            else:
                for j in range(1, CONTEXT + 1, 1):
                    temp_X.append(word2idx[wordsList[i + j]])

            X.append(temp_X)
            y_pred.append(word2idx[wordsList[i]])

    def _addToDictionaries(self, index, wordsList):
        for word in wordsList:
            if word not in word2idx:
                word2idx[word] = index
                idx2word[index] = word
                index += 1
        return index

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


twitterData = TwitterData("twitter-sentiment.csv")
train_loader = DataLoader(dataset=twitterData, batch_size=bsz, shuffle=True)

