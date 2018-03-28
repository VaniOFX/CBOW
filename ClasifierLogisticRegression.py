from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from dataReader import prepare_data


data, target = prepare_data("twitter-sentiment.csv")
train_sent, test_sent, train_lbl, test_lbl = train_test_split(data, target, test_size=1/7.0, random_state=0)
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_sent, train_lbl)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr.score(test_sent, test_lbl)))


data, target = prepare_data("twitter-sentiment-testset.csv")
y_pred = logisticRegr.predict(data)
f_pred = open("myPrediction.csv", 'w')
f_pred.write('ID\tSentiment\n')
ID = 1
for pred in y_pred:
    sentiment_pred = 'pos' if pred == 1 else 'neg'
    f_pred.write(str(ID)+','+sentiment_pred+'\n')
    ID += 1
