import torch
from torch.autograd import Variable
from CBOW import model, optimizer, loss_function
from loader import train_loader


def get_similarity(w1, w2, model):
    a = model.get_word_embedding(w1)
    b = model.get_word_embedding(w2)
    return (a.dot(b) / torch.norm(a) * torch.norm(b)).data.numpy()[0]


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

