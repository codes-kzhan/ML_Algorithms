import csv
import numpy as np

filename = "user-shows.txt"
Id = 500

def data(filename):
    for t, row in enumerate(open(filename)):
        row = row.strip().split()
        for i in range(len(row)):
            row[i] = float(row[i])
        yield t, np.array(row)

def get_vector(Id):
    for t, row in data(filename):
        if t == Id - 1:
            for i in range(len(row)):
                row[i] = float(row[i])
            return np.array(row)

def cosine(x, y):
    return float(np.dot(x, y)) / ((x.sum() ** 0.5) * (y.sum() ** 0.5))


class CollaborativeFiltering(object):

    def __init__(self, Id = Id, sim = cosine):
        # parameters
        self.vector = get_vector(Id)
        self.sim = sim
        self.rec = np.array([0.0] * len(self.vector))

    def accumulate(self, row):
        vector = self.vector
        sim = self.sim
        rec = self.rec

        similarity = sim(row, vector)
        rec += similarity * row

def solution(learner):
    for t, row in data(filename):
        if t != Id - 1:
            learner.accumulate(row)

if __name__ == '__main__':
    learner = CollaborativeFiltering()
    solution(learner)
