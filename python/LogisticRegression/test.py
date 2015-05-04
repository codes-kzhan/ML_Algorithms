import pandas as pd
from src.logistic import LogisticRegression

if __name__ == '__main__':
    filename = "dataset.csv"
    df = pd.read_csv(filename, header = 0)
    data = df.values
    y = data[:, -1]
    data = data[:, 0:-1]
    train = data[0:50000]
    ytrain = y[0:50000]
    test = data[50000:]
    ytest = y[50000:]
    clf = LogisticRegression()
    clf.train(train, ytrain)
    score = clf.score(test, ytest)
    print score
