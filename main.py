import torch
from torch.autograd import Variable
from CBOW import model, optimizer, loss_function
from loader import train_loader
import numpy as np
import matplotlib.pyplot as plt

EPOCH = 5
loss_data = []
iterations = []


def get_similarity(w1, w2, mod):
    a = mod.get_word_embedding(w1)
    b = mod.get_word_embedding(w2)
    return (a.dot(b) / torch.norm(a) * torch.norm(b)).data.numpy()[0]


def train():
    for epoch in range(1, EPOCH+1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 30 == 0:
                loss_data.append(loss.data[0])
                iterations.append(epoch * batch_idx)


plt.plot(iterations, loss_data)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()


word_emb_mat = model.embeddings.weight.numpy()
np.savetxt("word_emb_mat.txt", word_emb_mat)
