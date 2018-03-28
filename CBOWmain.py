import torch
from torch.autograd import Variable
from CBOW import model, optimizer, loss_function
from CBOWloader import train_loader
import numpy as np
import matplotlib.pyplot as plt
import time


EPOCH = 6


def get_similarity(w1, w2, mod):
    a = mod.get_word_embedding(w1)
    b = mod.get_word_embedding(w2)
    return (a.dot(b) / torch.norm(a) * torch.norm(b)).data.numpy()[0]


def train():
    iteration = 0
    for epoch in range(1, EPOCH+1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).cuda(), Variable(target).cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                iteration = iteration + 1
                if batch_idx % 30 == 0:
                    print("the loss is ", loss.data[0])
                loss_data.append(loss.data)
                iterations.append(iteration)


if __name__ == "__main__":
    start_time = time.time()
    loss_data = []
    iterations = []
    train()
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.plot(iterations, loss_data)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    word_emb_mat = model.embeddings.cpu().weight.data.numpy()
    print("Saving...")
    np.savetxt("word_emb_mat.txt", word_emb_mat)
    print(get_similarity('i', 'you', model))
    print(get_similarity('he', 'she', model))
    print(get_similarity('try', 'trying', model))
    print(get_similarity('sister', 'brother', model))
    print(get_similarity('happy', 'sad', model))
    print(get_similarity('happy', 'fun', model))
    print(get_similarity('her', 'him', model))
