import torch
import torch.nn as nn
from torch.autograd import Variable
from loader import word2idx, CONTEXT
import torch.nn.functional as F



EMDEDDING_DIM = 100
LEARNING_RATE = 0.00003

class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, context):
        super(CBOW, self).__init__()
        # self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.lin1 = nn.Linear(embedding_dim * context * 2, 128)
        # self.lin2 = nn.Linear(128, vocab_size)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lin1 = nn.Linear(context * 2 * embedding_dim, 512)
        self.lin2 = nn.Linear(512, vocab_size)

    def forward(self, inp):
        out = sum(self.embeddings(inp))
        out = out.view(1, -1)
        out = self.lin1(out)
        out = F.relu(out)
        out = self.lin2(out)
        out = F.log_softmax(out, dim=1)
        return out

    def get_word_embedding(self, word):
        word = Variable(torch.LongTensor([word2idx[word]]))
        return self.embeddings(word).view(1, -1)


model = CBOW(len(word2idx), EMDEDDING_DIM, CONTEXT).cuda()
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)






