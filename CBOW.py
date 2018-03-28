import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from CBOWloader import word2idx
import torch.nn.functional as F


EMDEDDING_DIM = 100
LEARNING_RATE = 0.0003


class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lin1 = nn.Linear(embedding_dim, vocab_size)
        # self.lin2 = nn.Linear(256, vocab_size)

    def forward(self, inp):
        out = self.embeddings(inp)
        out = torch.sum(out, dim=1)
        # print("raw input {}\nafter summing {}\n".format(inp.size(), out.size()))
        out = self.lin1(out)
        # out = F.relu(out)
        # out = self.lin2(out)
        return out

    def get_word_embedding(self, word):
        word = Variable(torch.LongTensor([word2idx[word]]))
        return self.embeddings(word)


model = CBOW(len(word2idx), EMDEDDING_DIM).cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

