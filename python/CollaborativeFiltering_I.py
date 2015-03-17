import csv
import numpy as np

filename = "user-shows.txt"
filename2 = "shows-user.txt"
Id = 500
mem = {}

def data1(filename):
    for t, row in enumerate(open(filename)):
        row = row.strip().split()
        for i in range(len(row)):
            row[i] = float(row[i])
        yield t, np.array(row)

def data2(filename2):
    for t, row in enumerate(open(filename2)):
        row = row.strip().split(',')
        for i in range(len(row)):
            row[i] = float(row[i])
        yield t, np.array(row)

def get_vector(Id):
    for t, row in data1(filename):
        if t == Id - 1:
            for i in range(len(row)):
                row[i] = float(row[i])
            return np.array(row)

def cosine(x, y, n, m):
    if (n, m) in mem:
        return mem[(n, m)]
    ret = float(np.dot(x, y)) / ((x.sum() ** 0.5) * (y.sum() ** 0.5))
    mem[(n, m)] = ret
    mem[(m, n)] = ret
    return ret


class CollaborativeFiltering(object):

    def __init__(self, Id = Id, sim = cosine):
        # parameters
        self.vector = get_vector(Id)
        self.sim = sim
        self.rec = np.array([0.0] * len(self.vector))

    def accumulate(self, row1, row2, n, m):
        vector = self.vector
        sim = self.sim
        rec = self.rec

        similarity = sim(row1, row2, n, m)
        rec[n] += similarity * vector[m]


def solution(learner):
    for t1, row1 in data2(filename2):
        print t1
        if t1 >= 100:
            break
        for t2, row2 in data2(filename2):
            learner.accumulate(row1, row2, t1, t2)

if __name__ == '__main__':
    learner = CollaborativeFiltering()
    solution(learner)
