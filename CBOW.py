from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import re
import io


EMDEDDING_DIM = 1000
CONTEXT = 2

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class TwitterData(Dataset):
    def __init__(self, filename, context):
        X = []
        y_pred = []
        word2idx = {'pad':0}
        idx2word = {0:'pad'}
        index = 1

        with io.open("twitter-sentiment.csv", "r", encoding="ISO-8859-1") as f:
            next(f)
            for line in f:
                _, _, sentence = line.split('\t')
                sentence = clean_str(sentence)
                wordsList = sentence.split()
                for word in wordsList:
                    if word not in word2idx:
                        word2idx[word] = index
                        idx2word[index] = word
                        index += 1
                for i in range(len(wordsList)):
                    temp_X = []
                    if i - context < 0:
                        pad_count = context - i
                        for j in range(pad_count):
                            temp_X.append(word2idx['pad'])
                        for j in range(context - pad_count, 0, -1):
                            temp_X.append(word2idx[wordsList[i - j]])
                    else:
                        for j in range(context, 0, -1):
                            temp_X.append(word2idx[wordsList[i - j]])

                    if i + context > len(wordsList) - 1:
                        pad_count = i + context - len(wordsList) + 1
                        for j in range(1, context - pad_count + 1):
                            temp_X.append(word2idx[wordsList[i + j]])
                        for j in range(pad_count):
                            temp_X.append(word2idx['pad'])

                    else:
                        for j in range(1, context + 1, 1):
                            temp_X.append(word2idx[wordsList[i + j]])
                    X.append(temp_X)
                    y_pred.append(i)

        self.x_data = X
        self.y_data = y_pred
        self.len = len(self.x_data)
        self.vocab_size = len(word2idx)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


twitterData = TwitterData("twitter-sentiment.cvs",CONTEXT)

train_loader = DataLoader(dataset=twitterData, batch_size=15, shuffle=True)


class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW,self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim,128)
        self.af1 = nn.ReLU()
        self.linear2 = nn.Linear(128, vocab_size)
        self.af2 = nn.LogSoftmax()


    def forward(self, input):
        embeds = sum(self.embeddings(input)).view(1, -1)
        out = self.linear1(embeds)
        out = self.af11(out)
        out = self.linear2(out)
        out = self.af2(out)
        return out

    def get_word_emdedding(self, word):
        return


model = CBOW(twitterData.vocab_size, EMDEDDING_DIM)

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)