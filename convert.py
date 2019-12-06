import fire
import numpy as np

def convert_to_incidence_list(name):
    with open(name+".tensors.dat", 'r') as f:
        vectors = [np.fromstring(s, sep='\t') for s in f.readlines()]
    with open(name+".labels.dat", 'r') as f:
        labels = [np.fromstring(s, dtype="int", sep='\t') for s in f.readlines()]
    with open(name+".sizes.dat", 'r') as f:
        sizes = [np.fromstring(s, dtype="int", sep='\t') for s in f.readlines()]

    tensors = []
    for t, s in zip(vectors, sizes):
        t = np.reshape(t.view(np.complex), s, order='F')
        tensors.append(t)
    return tensors, labels

def incidence_list_to_adjacency_list(labels):
    nt = len(labels)
    adjs = []
    ne = max([max(l) for l in labels])   # no. starts from 1
    table = np.zeros((nt, ne), dtype='bool')
    for it in range(nt):
        table[it, [i-1 for i in labels[it]]] = 1
    # find neighbors
    adjs = [np.zeros(len(l), dtype='int') for l in labels]
    for ie in range(ne):
        jts = np.where(table[:,ie])[0]
        if len(jts) != 2:
            print("can not find matching index! not a valid tensor network!")
        i, j = jts
        adjs[i][list(labels[i]).index(ie+1)] = j
        adjs[j][list(labels[j]).index(ie+1)] = i

    return adjs

"""
`name` is the prefix of data files, like `qft20`.
"""
def tensornetwork(name):
    tensors, labels = convert_to_incidence_list(name)
    labels = incidence_list_to_adjacency_list(labels)
    return tensors, labels

if __name__ == "__main__":
    fire.Fire(tensornetwork)
 
