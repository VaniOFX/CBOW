from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = []
target = []


train_sent, test_sent, train_lbl, test_lbl = train_test_split(data, target, test_size=1/7.0, random_state=0)
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_sent, train_lbl)



