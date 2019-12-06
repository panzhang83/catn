import torch
import numpy as np
import math
import sys
sys.path.append('..')
from torchsvd import SVD

svd = SVD.apply


class MPSNode:
    def __init__(self, tensor, index, neighbor, chi=32, cutoff=1.0e-15):
        self.tensor = tensor
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.index = index
        self.chi = chi
        self.cutoff = cutoff
        self.neighbor = neighbor
        self.type = "mps"
        self.mps = self.raw2mps(tensor)
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

    def raw2mps(self, tensor):
        if len(tensor.shape) == 0:  # scalar, a isolated node
            return []
        tensor0 = tensor.clone()
        shape = [1] + list(tensor.shape) + [1]
        if len(tensor.shape) == 1:  # degree 1, leaf
            return [tensor.reshape(shape)]
        order = len(tensor.shape)
        tensor = tensor.reshape(1, -1)
        mps = []
        for i in range(order - 1):
            dleft = tensor.shape[0]
            tensor = tensor.reshape(dleft * shape[i + 1], -1)
            [U, s, V] = svd(tensor)
            s_eff = s[s > self.cutoff]
            myd = min(len(s_eff), self.chi)
            s_eff = s_eff[:myd]
            U = U[:, :myd]
            V = V[:, :myd]
            s = torch.diag(s_eff)
            mps.append(U.reshape(dleft, shape[i + 1], myd))
            tensor = s @ V.t()
        mps.append(tensor.reshape(myd, shape[order], 1))
        self.cano = order - 1  # left canonical
        return mps

    def mps2raw(self, mps):
        if len(mps) < 2 and mps[0].numel() == 1:
            return mps[0]
        shape = [mps[0].shape[1]]
        tensor = mps[0].reshape(mps[0].shape[1], mps[0].shape[2])
        for i in range(1, len(mps)):
            shape = shape + [mps[i].shape[1]]
            a = mps[i]
            tensor = torch.einsum("ij,jkl->ikl", tensor, a).reshape(tensor.shape[0] * a.shape[1], a.shape[2])
        return tensor.reshape(shape)

    def move2tail(self, idx):
        """ 
        move idx to the end of mps 
        This must be careful: in this function, neighbors are not arranged.
        """
        if idx < 0:
            print("move2tail(): idx should be larger than 0")
            sys.exit(0)
        if idx == len(self.mps) - 1:
            self.cano_to(-1)
            return
        for i in range(idx, len(self.mps) - 1):
            self.swap(i, i + 1)
        self.cano_to(-1)

    def move(self, a, b):
        """
        move the tensor from index a to index b by swapping consecutive tensors
        """
        if a == b:
            return
        if a < 0 or b < 0:
            print("move2tail(): idx should be larger than 0")
            sys.exit(0)
        if b > a:
            b = b
            for i in range(a, b):
                self.swap(i, i + 1)
        else:
            b = b
            for i in range(a, b, -1):
                self.swap(i, i - 1)

    def cano_to(self, idx):
        """
        move canonical position to i
        """
        if idx == -1:
            idx = len(self.mps) - 1
        if self.cano == idx:  # there is nothing to do
            return
        if self.cano < idx:
            for i in range(self.cano, idx):
                dl = self.mps[i].shape[0]
                d = self.mps[i].shape[1]
                # Q, R = torch.qr(self.mps[i].reshape(dl * d, -1))
                [U, s, V] = svd(self.mps[i].reshape(dl * d, -1))
                #Q = U @ torch.diag(torch.sqrt(s))
                #R = torch.diag(torch.sqrt(s)) @ V.t()
                Q = U
                R = torch.diag(s) @ V.t()
                self.mps[i] = Q.reshape(dl, d, -1)
                self.mps[i + 1] = torch.einsum("ij,jab->iab", R, self.mps[i + 1])
                self.cano = i + 1
        else:
            for i in range(self.cano, idx, -1):
                dr = self.mps[i].shape[2]
                d = self.mps[i].shape[1]
                # Q, R = torch.qr(self.mps[i].reshape(-1, d * dr).t())
                [U, s, V] = svd(self.mps[i].reshape(-1, d * dr).t())
                #Q = U @ torch.diag(torch.sqrt(s))
                #R = torch.diag(torch.sqrt(s)) @ V.t()
                Q = U
                R = torch.diag(s) @ V.t()
                self.mps[i] = Q.t().reshape(-1, d, dr)
                self.mps[i - 1] = torch.einsum("abc,ci->abi", self.mps[i - 1], R.t())
                self.cano = i - 1

    def swap(self, i, j):
        """
        swap index i and index j in mps, i and j must be consecutive indices
        Assuming that canonical form is maintained.
        Default direction is i \to j, that is the canonical position will be j after swap
        The canonicalization is maintained.
        """
        if j < 0 or j > len(self.mps):
            return
        if self.cano != i and self.cano != j:
            self.cano_to(i if abs(self.cano - i) < abs(self.cano - j) else j)

        if abs(i - j) != 1:
            print("swap(): i and j must be consecutive indices, there must be something wrong")
            sys.exit(3)

        if i < j:
            tl = self.mps[i]
            tr = self.mps[j]
        else:
            tl = self.mps[j]
            tr = self.mps[i]

        d0 = tl.shape[0]
        d1 = tr.shape[1]
        d2 = tl.shape[1]
        d3 = tr.shape[2]
        mat = torch.einsum("ijk,kab->iajb", tl, tr).reshape(d0 * d1, d2 * d3)  # swaped
        #print(mat)
        [U, s, V] = svd(mat)
        s_eff = s[s > self.cutoff]
        #print(s_eff)
        myd = min(len(s_eff), self.chi)
        if myd == 0:
            print("Warning in swap(), probably a zero matrix is encountered !!! myd=", myd)
            sys.exit(-7)
        s_eff = s_eff[:myd]
        U = U[:, :myd]
        V = V[:, :myd]
        s = torch.diag(s_eff)
        if i < j:  # going right
            V = s @ V.t()
            self.mps[i] = U.reshape(d0, d1, myd)
            self.mps[j] = V.reshape(myd, d2, d3)
        else:  # going left
            U = U @ s
            self.mps[j] = U.reshape(d0, d1, myd)
            self.mps[i] = V.t().reshape(myd, d2, d3)
        self.cano = j

    def shape(self, idx=math.inf):
        if idx == math.inf:
            if len(self.mps) == 1:
                return [1]
            else:
                return [i.shape[1] for i in self.mps]
        else:
            return self.mps[idx].shape[1]

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
        if not cross:
            self.move(idx2, idx1 + 1)
        else:
            self.move(idx2, idx1)
        self.cano_to(idx1)
        self.mps[idx1] = torch.einsum("ijk,kab->ijab", self.mps[idx1], self.mps[idx1 + 1]).reshape(
            self.mps[idx1].shape[0], -1, self.mps[idx1 + 1].shape[2])
        self.mps.pop(idx1 + 1)
        self.cano_to(idx1)

    def logdim(self, idx=math.inf):
        """ return log of number of elements of the raw tensor"""
        try:
            if len(self.mps) == 0:
                return 0
        except:
            return 0
        if idx != math.inf:
            return math.log2(self.mps[idx].shape[1])
        else:
            return torch.log2(
                torch.tensor([i.shape[1] for i in self.mps], dtype=self.dtype, device=self.device)).sum().item()

    def order(self):
        """ return order of the tensor """
        try:
            if len(self.mps) == 0:
                return 0
        except:
            return 0
        return len(self.mps)

    def move2tail_neighbor(self, idx):
        self.neighbor = list(self.neighbor[:idx]) + list(self.neighbor[idx + 1:]) + [self.neighbor[idx]]

    def move2head(self, idx):
        self.move(idx, 0)  # notice that j's neighbors are not modified !
        self.cano_to(0)

    def eat(self, nodej, idx, idxi):
        """ 
        Eat node j, that is contract idx of self to idxi of nodej, appending all neighbors of j to itself 
        TODO:
            1. Moving to end and Moving to begin could be heavy if the position is not good enough. Considering reverse the whole chain before moving.
        """
        if len(self.mps) == 1:  # the node i is a leaf, according to the regulation introduced in contraction(), node j must be no larger than node i, so j must be a leaf as well
            assert self.mps[0].shape[0] == 1 and self.mps[0].shape[2] == 1 and len(nodej.mps) == 1 and \
                   nodej.mps[0].shape[0] == 1 and nodej.mps[0].shape[2] == 1
            lognorm = torch.log(
                self.mps[0].reshape(1, self.mps[0].shape[1]) @ nodej.mps[0].reshape(nodej.mps[0].shape[1], 1)).reshape(-1)
            self.mps = []
            return lognorm

        self.move2tail(idx)
        mati = self.mps[-1].reshape(self.mps[-1].shape[:-1])

        if len(nodej.mps) == 1:  # node i is not a leaf, j is a leaf
            assert nodej.mps[0].shape[0] == 1 and nodej.mps[0].shape[2] == 1
            assert (self.cano == len(self.mps) - 1)
            tensorj = nodej.mps[0]
            matj = tensorj.reshape(tensorj.shape[1], 1)
            mat = mati @ matj
            new_tensor = torch.einsum("ijk,ka->ija", self.mps[-2], mat)
            norm = new_tensor.norm()
            self.mps[-2] = new_tensor / norm
            self.cano = self.cano - 1
            self.mps.pop(-1)
            return torch.log(norm)

        nodej.move2head(idxi)
        matj = nodej.mps[0].reshape(nodej.mps[0].shape[1:])
        mat = mati @ matj
        self.mps[-2] = torch.einsum("ijk,ka->ija", self.mps[-2], mat)
        self.mps.pop(-1)
        self.cano = len(self.mps) - 1
        for a in range(1, len(nodej.mps)):
            self.mps.append(nodej.mps[a])
        self.cano_to(-1)
        norm = self.mps[self.cano].norm()
        self.mps[self.cano] = self.mps[self.cano] / norm
        if norm <= self.cutoff:
            return 0
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
        lognorm = torch.tensor(0)
        if len(self.mps) == 0:
            return 0
        for i in self.mps:
            if i.numel() == 0:
                continue
            norm = i.norm()
            i = i / norm
            lognorm = lognorm + torch.log(norm)
        return lognorm

    def clear(self):
        self.mps = []
        self.neighbor = []
