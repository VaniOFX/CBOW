from ClasifierNN import model, optimizer, loss_function
from ClasifierLoader import train_loader
from torch.autograd import Variable


for batchidx, (x,y) in enumerate(train_loader):
    model.train()
    data, target = Variable(x).cuda(), Variable(y).cuda()
    optimizer.zero_grad()
    output = model(x)
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()