from ClasifierNN import model, optimizer, loss_function
from ClasifierLoader import train_loader
from torch.autograd import Variable


for batchidx, (x, y) in enumerate(train_loader):
    model.train()
    data, target = Variable(x).cuda(), Variable(y).cuda()
    optimizer.zero_grad()
    output = model(data)
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
    if batchidx % 30 == 0:
        print("the loss is", loss.data[0])
