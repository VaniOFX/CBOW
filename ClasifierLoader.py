from torch.utils.data import DataLoader, Dataset
from dataReader import prepare_data
import torch

bsz = 64

class MyDataSet(Dataset):

    def __init__(self, filename):
        X, y = prepare_data(filename)
        self.x_data = torch.Tensor(X)
        self.y_data = torch.LongTensor(y)
        self.len = len(X)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


twitterData = MyDataSet('twitter-sentiment.csv')
train_loader = DataLoader(dataset=twitterData, batch_size=bsz, shuffle=True)