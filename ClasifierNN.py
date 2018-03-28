import torch
import torch.nn as nn
import torch.nn.functional as F


LEARNING_RATE = 0.000003


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(100, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, input):
        out = F.relu(self.linear1(input))
        out = self.linear2(out)
        return out


model = Model().cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

