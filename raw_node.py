import torch
import numpy as np
import math
import sys


class Node:
    def __init__(self, tensor, index, neighbor, cutoff=1.0e-15):
        self.tensor = tensor
        self.index = index
        self.cutoff = cutoff
        self.neighbor = neighbor
        self.type = "raw"
        self.cano = 0  # position of canonicalization

    def find_neighbor(self, j):
        re = np.argwhere(self.neighbor == j)
        if len(re) == 0:
            return -1
        elif len(re) > 1:
            print("Something wrong in find_neighbor: there are two positions storing the same neighbor")
            sys.exit(3)
        else:
            return re[0][0]

    def shape(self, idx=math.inf):
        if idx == math.inf:
            if not isinstance(self.tensor, list):
                return list(self.tensor.shape)
            else:
                return []
        else:
            return self.tensor.shape[idx]

    def merge(self, j, cross=False):
        """ 
        merge two identitical neighbors of i
        """
        idxj = np.argwhere(self.neighbor == j)
        shape = self.shape()
        if idxj.size != 2:
            print("there is nothing to do in self.merge() !")
            sys.exit(4)
            return
        idx1 = idxj[0][0]
        idx2 = idxj[1][0]
        self.neighbor = np.delete(self.neighbor, idx2)
        seq = [a for a in range(self.order())]
        seq.pop(idx2)
        if not cross:
            seq.insert(idx1 + 1, idx2)
        else:
            seq.insert(idx1, idx2)
        shape = np.array(shape)
        shape = shape[seq]
        shape[idx1] = shape[idx1] * shape[idx1 + 1]
        shape = np.delete(shape, idx1 + 1)
        self.tensor = self.tensor.permute(seq)
        self.tensor = self.tensor.reshape(-1)
        self.tensor = self.tensor.reshape(list(shape))

    def logdim(self, idx=math.inf):
        """ return log of number of elements of the raw tensor"""
        try:
            if len(self.tensor) == 0:
                return -1
        except:
            return -1
        if idx != math.inf:
            return math.log2(self.tensor.shape[idx])
        else:
            return torch.log2(
                torch.tensor(self.tensor.shape, dtype=self.tensor.dtype, device=self.tensor.device)).sum().item()

    def order(self):
        """ return order of the tensor """
        try:
            if len(self.tensor) == 0:
                return 0
        except:
            return 0
        return len(self.tensor.shape)

    def unfolding(self, idx):
        """ return a unfolded matrix with idx at the secon dimension """
        seqi = [a for a in range(self.order())]
        seqi.pop(idx)
        seqi.append(idx)
        return self.tensor.permute(seqi).reshape(-1, self.shape(idx))

    def restore_from_matrix(self, mat, idx):
        """ store the given unfolded matrix mat to self.tensor.
            at the same time restoring the shape of the tensor"""
        shapei = list(self.tensor.shape[:idx]) + list(self.tensor.shape[idx + 1:]) + [mat.shape[1]]
        self.tensor = mat.reshape(shapei)
        self.neighbor = list(self.neighbor[:idx]) + list(self.neighbor[idx + 1:]) + [self.neighbor[idx]]

    def eat(self, nodej, idx, idxi):
        """ 
        Eat node j, that is contract idx of self to idxi of nodej, appending all neighbors of j to itself 
        TODO:
            1. Moving to end and Moving to begin could be heavy if the position is not good enough. Considering reverse the whole chain before moving.
        """
        seqi = [a for a in range(self.order())]
        seqi.pop(idx)
        seqi.append(idx)
        self.tensor = self.tensor.permute(seqi)
        shapei = [a for a in self.tensor.shape]
        self.tensor = self.tensor.reshape(-1, self.shape(-1))

        shapej = [a for a in nodej.tensor.shape[:idxi]] + [a for a in nodej.tensor.shape[idxi + 1:]]
        shapej = list(nodej.tensor.shape[:idxi]) + list(nodej.tensor.shape[idxi + 1:])
        matj = nodej.unfolding(idxi)
        self.tensor = (self.tensor @ matj.t()).reshape(shapei[:-1] + shapej)
        norm = self.tensor.norm()
        self.tensor = self.tensor / norm
        matj=torch.ones(1)
        return torch.log(norm)

    def add_neighbor(self, n, pos=-1):
        if pos != -1:
            self.neighbor = np.insert(self.neighbor, pos, n)
        else:
            self.neighbor = np.append(self.neighbor, n)

    def delete_neighbor(self, n):
        idx = np.argwhere(self.neighbor == n)
        self.neighbor = np.delete(self.neighbor, idx)
        return idx[0][0]

    def lognorm(self):
        if self.tensor.numel() == 0:
            return torch.tensor(0)
        else:
            return torch.log(self.tensor.norm())

    def clear(self):
        self.tensor = torch.ones(1)
        self.neighbor = []
