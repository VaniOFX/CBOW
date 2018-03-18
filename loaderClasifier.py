from torch.utils.data import DataLoader, Dataset
from cleanData import clean_str
import io


bsz = 35

class MyDataSet(Dataset):

    def __init__(self, filename):
        X = []
        y_pred = []
        with io.open(filename, "r", encoding="ISO-8859-1") as f:
            next(f)
            for line in f:
                ID, label, sentence = line.split('\t')
                label_idx = 1 if label == 'pos' else 0  # 1 for pos and 0 for neg

                rev = []
                rev.append(sentence.strip())
                orig_rev = clean_str(" ".join(rev))

                x_temp = self.convertToVector(orig_rev)
                X.append(x_temp)
                y_pred.append(label_idx)

        self.x_data = X
        self.y_data = y_pred

    def convertToVector(self, rev):
        pass

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


twitterData = MyDataSet("twitter-sentiment.cvs")
train_loader = DataLoader(dataset=twitterData, batch_size=bsz, shuffle=True)