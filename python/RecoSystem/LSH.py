import numpy as np
import csv
from collections import defaultdict
import scipy.spatial
import doctest


class CosineNN(object):

    BLOCK_SIZE, NUM_BLOCKS = 16, 500
    SIG_LENGTH = BLOCK_SIZE * NUM_BLOCKS

    def __init__(self, vector_len):
        self.col_vecs = {}
        self.signatures = {}
        self.random_vector_family = read_hash_family()

        # nn_index[(which_block, block_integer_value)] = list of m-ids in bucket
        self.nn_index = defaultdict(list)

    def index(self, iid, col):
        """
        Indexes item with ID `iid', column vector `col' into data structure.
        """
        # Form the column vector of the utility matrix for m
        sig = self.signature_of(col)

        self.signatures[iid] = sig
        self.col_vecs[iid] = col

        # Index this signature
        for block_num in range(CosineNN.NUM_BLOCKS):
            block_val = extract_block(sig, block_num)
            self.nn_index[(block_num, block_val)].append(iid)

    def query(self, iid):
        sig = self.signatures[iid]
        resultset = set()
        for block_num in range(CosineNN.NUM_BLOCKS):
            block_val = extract_block(sig, block_num)
            resultset.update(self.nn_index[(block_num, block_val)])
        return resultset


    def query_with_dist(self, iid):
        maybe_neighbours = self.query(iid)
        with_dist = [(niid, self.cosine_dist_between(iid, niid))
                     for niid in maybe_neighbours if niid != iid]
        return with_dist

    def signature_of(self, vec):
        """Takes a numpy vector of length U and produces its LSH."""
        sketch = self.random_vector_family * vec  # This is a matrix product
        num = 0
        # Generate the signature (an integer)
        # TODO: find a way of vectorising this loop.
        for i in range(CosineNN.SIG_LENGTH):
            if sketch[i, 0] >= 0:
                num |= (1 << i)
        return num

    def cosine_dist_between(self, iid1, iid2):
        # Requires .todense() or else 'dimension missmatch'
        return scipy.spatial.distance.cosine(self.col_vecs[iid1].todense(),
                                             self.col_vecs[iid2].todense())


def extract_block(sig, block_num):
    return (sig >> (block_num*CosineNN.BLOCK_SIZE)) % (1 << CosineNN.BLOCK_SIZE)

def create_hash_family(vector_len):
    # CosineNN.SIG_LENGTH * vector_len matrix where each column is a vector
    # corresponding to a random hyperplane in the family. All entries are in
    # [-0.5, 0.5).
    random_vector_family = np.matrix(
        np.random.rand(CosineNN.SIG_LENGTH, vector_len) - 0.5)
    filename = open("hash_function.csv", "wb")
    open_file_object = csv.writer(filename)
    for vec in random_vector_family:
        vec = np.array(vec)[0]
        row = [str(i) for i in vec]
        open_file_object.writerow(row)

def read_hash_family(filename = "hash_function.csv"):
    openfile = open(filename)
    random_vector_family = []
    for row in openfile:
        row = row.strip().split(',')
        floatRow = [float(i) for i in row]
        random_vector_family.append(floatRow)
    return np.matrix(random_vector_family)
        
    
if __name__ == '__main__':
    vec1 = np.mat(np.random.rand(93)).T
    vec2 = np.mat(np.random.rand(93)).T
    vec3 = np.mat(np.random.rand(93)).T
    vec4 = np.mat(np.random.rand(93)).T
    tt = CosineNN(93)
    tt.index(1, vec1)
    tt.index(2, vec2)
    tt.index(3, vec3)
    tt.index(4, vec4)
    
