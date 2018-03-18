import torch
import torch.nn as nn
from torch.autograd import Variable
from loader import word2idx


EMDEDDING_DIM = 1000
LEARNING_RATE = 0.001

class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
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
        word = Variable(torch.Tensor([word2idx[word]]))
        return self.embeddings(word).view(1, -1)


model = CBOW(len(word2idx), EMDEDDING_DIM)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)






