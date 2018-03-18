import torch
import torch.nn as nn

LEARNING_RATE = 0.01


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(None, 128)
        self.af1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 2)
        self.af2 = nn.LogSoftmax()

    def forward(self, input):
        out = self.af1(self.linear1(input))
        out = self.af2(self.linear2(out))
        return out


model = Model()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

def train():
    return
