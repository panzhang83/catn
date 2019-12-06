import time
import numpy as np
import math
import sys
from npsvd import svd,rsvd

class MPSNode:
    def __init__(self, tensor, index, neighbor,chi=32,cutoff=1.0e-15,norm_method=1,svdopt=True,swapopt=True,verbose=0):
        self.tensor = tensor
        self.svdopt=svdopt
        self.swapopt = swapopt
        self.dtype = tensor.dtype
        self.norm_method=norm_method
        self.index = index
        self.chi=chi
        self.cutoff=cutoff
        self.neighbor = neighbor
        self.type="mps"
        self.mps = self.raw2mps(tensor)
        self.cano=0 # position of canonicalization

    def find_neighbor(self,j):
        re=np.argwhere(self.neighbor==j)
        if(len(re) == 0):
            return -1
        elif(len(re)>1):
            print("Something wrong in find_neighbor: there are two positions storing the same neighbor")
            sys.exit(3)
        else:
            return re[0][0]


    def raw2mps(self,tensor):
        if(len(tensor.shape)==0): # scalar, a isolated node
            return []
        shape = [1]+list(tensor.shape)+[1]
        if(len(tensor.shape)==1): # degree 1, leaf
            return [tensor.reshape(shape)]
        order = len(tensor.shape)
        tensor = tensor.reshape(1,-1)
        mps=[]
        for i in range(order-1):
            dleft = tensor.shape[0]
            tensor = tensor.reshape(dleft*shape[i+1],-1)
            [U,s,V] = svd(tensor)
            s_eff = s[s>self.cutoff]
            myd = min(len(s_eff),self.chi)
            s_eff=s_eff[:myd]
            U=U[:,:myd]
            V=V[:,:myd]
            s=np.diag(s_eff)
            mps.append(U.reshape(dleft,shape[i+1],myd))
            tensor = s@V.T
        mps.append(tensor.reshape(myd,shape[order],1))
        self.cano = order-1 # left canonical
        return mps

    def mps2raw(self,mps):
        if(len(mps)<2 and mps[0].numel()==1):
            return mps[0]
        shape = [mps[0].shape[1]]
        tensor = mps[0].reshape(mps[0].shape[1],mps[0].shape[2])
        for i in range(1,len(mps)):
            shape = shape+[mps[i].shape[1]]
            a = mps[i]
            tensor = np.einsum("ij,jkl->ikl",tensor,a).reshape(tensor.shape[0]*a.shape[1],a.shape[2])
        return tensor.reshape(shape)

    def move2tail(self,idx):
        """ 
        move idx to the end of mps 
        This must be careful that in this function, neighbors are not arranged.

        """
        error = 0
        if(idx<0):
            print("move2tail(): idx should be larger than 0")
            sys.exit(0)
        if(idx == len(self.mps)-1):
            self.cano_to(-1)
            return error
        for i in range(idx,len(self.mps)-1):
            error = error + self.swap(i,i+1)
        self.cano_to(-1)
        return error

    def move(self,a,b):
        """
        move the tensor from index a to index b by swapping consecutive tensors
        """
        error = 0
        if(a==b):
            return error
        if(a<0 or b<0):
            print("move2tail(): idx should be larger than 0")
            sys.exit(0)
        if(b>a):
            b = b
            for i in range(a,b):
                error = error + self.swap(i,i+1)
        else:
            b = b
            for i in range(a,b,-1):
                error = error + self.swap(i,i-1)
        return error




    def cano_to(self,idx):
        """
        move canonical position to i
        """
        if(idx == -1):
            idx = len(self.mps)-1
        if(self.cano == idx): # there is nothing to do
            return
        if(self.cano < idx):
            for i in range(self.cano,idx):
                dl = self.mps[i].shape[0]
                d = self.mps[i].shape[1]
                #Q,R = torch.qr(self.mps[i].reshape(dl * d,-1))
                #U,s,V = torch.svd(self.mps[i].reshape(dl * d,-1))
                U,s,V = svd(self.mps[i].reshape(dl * d,-1))
                #Q=U
                #R=torch.diag(s)@V.t()
                seff = s[s>self.cutoff]
                myd = seff.shape[0]
                if(myd==0):
                    myd = U.shape[1]
                else:
                    s = seff
                Q=U[:,:myd]
                R=np.diag(s)@(V[:,:myd]).T
                self.mps[i] = Q.reshape(dl,d,-1)
                self.mps[i+1] = np.einsum("ij,jab->iab",R,self.mps[i+1])
                self.cano = i+1
        else:
            for i in range(self.cano,idx,-1):
                dr = self.mps[i].shape[2]
                d = self.mps[i].shape[1]
                #Q,R = torch.qr(self.mps[i].reshape(-1,d*dr).t())
                #U,s,V = torch.svd(self.mps[i].reshape(-1,d*dr).t())
                U,s,V = svd(self.mps[i].reshape(-1,d*dr).T)
                #Q=U
                #R=torch.diag(s)@V.t()
                seff = s[s>self.cutoff]
                myd = seff.shape[0]
                if(myd==0):
                    myd = U.shape[1]
                else:
                    s = seff
                Q=U[:,:myd]
                R=np.diag(s)@(V[:,:myd].T)

                self.mps[i] = Q.T.reshape(-1,d,dr)
                self.mps[i-1] = np.einsum("abc,ci->abi",self.mps[i-1],R.T)
                self.cano = i-1
        return 0

    def left_canonical(self):
        self.cano = 0
        self.cano_to(-1)

    def compress(self):
        """
        Compress the whole mps.
        First, do left canonicalization to move self.cano to -1.
        Second, do two-site merging-splitting, for moving self.cano back to 0.
        """
        error = 0
        if len(self.mps) == 0:
            return error
        self.left_canonical() # now self.cano is at the bottom (right)
        for j in range( len(self.mps)-1,0,-1):
            i = j-1
            tl = self.mps[i]
            tr = self.mps[j]

            d0 = tl.shape[0]
            d1 = tl.shape[1]

            d2 = tr.shape[1]
            d3 = tr.shape[2] # notice the difference to self.swap()
            mat = np.einsum("ijk,kab->ijab",tl,tr).reshape(d0*d1,d2*d3)  # notice the difference to self. swap()
            [U,s,V] = svd(mat)
            s_eff = s[s>self.cutoff]
            myd = min(len(s_eff),self.chi)
            if(myd == 0):
                print("Warning in swap(), probably a zero matrix is encountered !!! myd=",myd)
                sys.exit(-8)
            s_eff=s_eff[:myd]
            error = error + s[myd:].sum()
            U=U[:,:myd]
            V=V[:,:myd]
            s=np.diag(s_eff)
            U = U@s
            self.mps[i] = U.reshape(d0,d1,myd)
            self.mps[j] = V.T.reshape(myd,d2,d3)
        self.cano = 0

    def compress_opt(self):
        """
        Compress the whole mps.
        First, do left canonicalization to move self.cano to -1.
        Second, do two-site merging-splitting, for moving self.cano back to 0.
        Do qr before SVD
        """
        error = 0
        if len(self.mps) == 0:
            return error
        self.left_canonical() # now self.cano is at the bottom (right)
        for j in range( len(self.mps)-1,0,-1):
            i = j-1
            tl = self.mps[i]
            tr = self.mps[j]

            d0 = tl.shape[0]
            d1 = tl.shape[1]

            d2 = tr.shape[1]
            d3 = tr.shape[2] # notice the difference to self.swap()

            dd = tl.shape[2]
            assert(dd == tr.shape[0])
#            mat = torch.einsum("ijk,kab->ijab",tl,tr).reshape(d0*d1,d2*d3)  # notice the difference to self. swap()
            matl = tl.reshape(d0*d1,dd)
            matr = tr.reshape(dd,d2*d3)

#            flag=False
            #if(matl.shape[0]*matr.shape[1] > dd*dd):
#            if(1==1):
#                flag=True
#                Ql,Rl = np.linalg.qr(matl)
#                Qr,Rr = np.linalg.qr(matr.T)
#                mat = Rl@Rr.T
#            else:
#                mat = matl@matr

            flag_left = False
            flag_right = False

            if(matl.shape[0] > matl.shape[1]):
                flag_left = True
                Ql,Rl = np.linalg.qr(matl)
            else:
                Rl = matl

            if(matr.shape[0] < matr.shape[1]):
                flag_right = True
                Qr,Rr = np.linalg.qr(matr.T)
            else:
                Rr = matr.T

            mat = Rl@Rr.T

            [U,s,V] = svd(mat)
            s_eff = s[s>self.cutoff]
            if(len(s_eff) == 0):
                s_eff = s[:1]
            myd = min(len(s_eff),self.chi)
            if(myd == 0):
                print("Warning in swap(), probably a zero matrix is encountered !!! myd=",myd)
                sys.exit(-8)
            s_eff=s_eff[:myd]
            error = error + s[myd:].sum()
            U=U[:,:myd]
            V=V[:,:myd]
            s=np.diag(s_eff)
            U = U@s
#            if flag:
#                U = Ql @ U
#                V =  Qr @ V
            if flag_left:
                U = Ql @ U
            if flag_right:
                V =  Qr @ V


            self.mps[i] = U.reshape(d0,d1,myd)
            self.mps[j] = V.T.reshape(myd,d2,d3)
#            print("correct after swap:, error=",self.check_mps())
        self.cano = 0



    def swap(self,i,j):
        """
        swap index i and index j in mps, i and j must be consecutive indices
        Assuming that canonical form is maintained.
        Default direction is i \to j, that is the canonical position will be j after swap
        The canonicalization is maintained.
        """
        error = 0
#        sys.stdout.write(" swap %d %d cano=%d "%(i,j,self.cano));sys.stdout.flush()
        if(j<0 or j>len(self.mps)):
            return
#        print("in swap(), move cano")
        if(self.cano != i and self.cano != j):
           self.cano_to( i if abs(self.cano-i)<abs(self.cano-j) else j)

        if(abs(i-j) != 1):
            print("swap(): i and j must be consecutive indices, there must be something wrong")
            sys.exit(3)

        if(i<j):
            tl = self.mps[i]
            tr = self.mps[j]
        else:
            tl = self.mps[j]
            tr = self.mps[i]

        d0 = tl.shape[0]
        d1 = tr.shape[1]
        d2 = tl.shape[1]
        d3 = tr.shape[2]
        mat = np.einsum("ijk,kab->iajb",tl,tr).reshape(d0*d1,d2*d3) # swaped
        if( self.swapopt and ((mat.shape[0] > 7000  and mat.shape[1] > 7000) or (mat.shape[0] > 20000  or mat.shape[1] > 20000)) ):
            [U,s,V] = rsvd(mat,self.chi,10,10)
        else:
            [U,s,V] = svd(mat)
        s_eff = s[s>self.cutoff]
        if(len(s_eff) == 0):
            s_eff = s[:1]
        myd = min(len(s_eff),self.chi)
        if(myd == 0):
            print("Warning in swap(), probably a zero matrix is encountered !!! myd=",myd)
            sys.exit(-7)
        s_eff=s_eff[:myd]
        error = error + s[myd:].sum()
        U=U[:,:myd]
        V=V[:,:myd]
        s=np.diag(s_eff)
        if(i<j):#going right
            V = s@V.T
            self.mps[i] = U.reshape(d0,d1,myd)
            self.mps[j] = V.reshape(myd,d2,d3)
        else:# going left
            U = U@s
            self.mps[j] = U.reshape(d0,d1,myd)
            self.mps[i] = V.T.reshape(myd,d2,d3)
        self.cano = j
        return error

    def shape(self,idx=math.inf):
        if(idx == math.inf):
            if(len(self.mps)==1):
                return [1]
            else:
                return [i.shape[1] for i in self.mps]
        else:
            return self.mps[idx].shape[1]

    def merge(self,j,cross=False):
        """ 
        merge two identitical neighbors of i
        """
        error = 0
        idxj = np.argwhere(self.neighbor == j)
        shape = self.shape()
        if(idxj.size != 2):
            print("there is nothing to do in self.merge() !")
            sys.exit(4)
            return
        idx1 = idxj[0][0]
        idx2 = idxj[1][0]
        self.neighbor = np.delete(self.neighbor,idx2)
        if(not cross):
            error = self.move(idx2,idx1+1)
        else:
            error = self.move(idx2,idx1)
        self.cano_to(idx1)
        self.mps[idx1] = np.einsum("ijk,kab->ijab",self.mps[idx1],self.mps[idx1+1]).reshape(self.mps[idx1].shape[0],-1,self.mps[idx1+1].shape[2])
        self.mps.pop(idx1+1)
        self.cano_to(idx1)
        return error

    def logdim(self,idx=math.inf):
        """ return log of number of elements of the raw tensor"""
        try:
            if(len(self.mps)==0):
                return 0
        except:
            return 0
        if(idx != math.inf):
            return math.log2(self.mps[idx].shape[1])
        else:
            return np.log2(np.array([i.shape[1] for i in self.mps]).astype(np.float64)).sum().item() #**************************

    def order(self):
        """ return order of the tensor """
        try:
            if len(self.mps)==0:
                    return 0
        except:
            return 0
        return len(self.mps)

    def move2tail_neighbor(self,idx):
        self.neighbor = list(self.neighbor[:idx]) + list(self.neighbor[idx+1:])+[self.neighbor[idx]]

    def move2head(self,idx):
        error = self.move(idx,0) # notice that j's neighbors are not modified !
        self.cano_to(0)
        return error

    def eat(self,nodej,idx, idxi):
        """ 
        Eat node j, that is contract idx of self to idxi of nodej, appending all neighbors of j to itself 
        TODO:
            1. Moving to end and Moving to begin could be heavy if the position is not good enough. Considering reverse the whole chain before moving.
        """
        error = 0
        if( len(self.mps) == 1): # the node i is a leaf, according to the regulation introduced in contraction(), node j must be no larger than node i, so j must be a leaf as well
#            print("two leaves")
            assert self.mps[0].shape[0] == 1 and self.mps[0].shape[2] == 1 and len(nodej.mps) == 1 and nodej.mps[0].shape[0] == 1 and nodej.mps[0].shape[2] == 1
            result = self.mps[0].reshape(1,self.mps[0].shape[1]) @ nodej.mps[0].reshape( nodej.mps[0].shape[1],1)
#            if(norm<0):
#                print("the result is smaller than 0",norm)
#                return 0,norm.item()
#            print("dot of two vectors:",result,np.linalg.norm(result),np.linalg.norm(result)**2)
#            print("dot of two vectors:",result,np.abs(result),np.abs(result)**2)

            lognorm = math.log(np.abs(result))
            self.mps = []
            return lognorm,0,result/np.abs(result)


#        idx_i_in_j=np.argwhere(nodej.neighbor == self.index)[0][0]
#        for l in range(len(nodej.neighbor)):
#            if l != idx_i_in_j:
#

        error = error + self.move2tail(idx)
        mati = self.mps[-1].reshape(self.mps[-1].shape[:-1])

        if( len(nodej.mps) == 1): # node i is not a leaf, j is a leaf
            assert nodej.mps[0].shape[0] == 1 and nodej.mps[0].shape[2] == 1
            assert(self.cano == len(self.mps)-1)
            tensorj = nodej.mps[0]
            matj = tensorj.reshape(tensorj.shape[1],1)
            mat = mati @ matj
            new_tensor = np.einsum("ijk,ka->ija",self.mps[-2],mat)

            if(self.norm_method == 1):
                norm = np.linalg.norm(new_tensor)
            elif self.norm_method == 2:
                norm = np.abs(new_tensor).max()
            elif self.norm_method == 0:
                norm = np.array(1)
            else:
                print("in eat(), norm_method not understood")
                sys.exit(-8)

            #norm = np.linalg.norm(new_tensor)
            self.mps[-2] = new_tensor / norm
            self.cano = self.cano - 1
            self.mps.pop(-1)
            return math.log(norm),error,1

        error = error + nodej.move2head(idxi)
        matj = nodej.mps[0].reshape(nodej.mps[0].shape[1:])

        mat =  mati @ matj
        if(len(self.mps) > 1):
            self.mps[-2] = np.einsum("ijk,ka->ija",self.mps[-2],mat)
            self.mps.pop(-1)
            self.cano = len(self.mps)-1
        else:
            print("Warning: this should never happen in eat()")
            sys.exit(-6)
        for  a in range(1,len(nodej.mps)):
            self.mps.append(nodej.mps[a])
        self.cano_to(-1)
        if(self.norm_method == 1):
            norm = np.linalg.norm(self.mps[self.cano])
        elif self.norm_method == 2:
            norm = np.abs(self.mps[self.cano]).max()
        elif self.norm_method == 0:
            norm = np.array(1).astype(self.dtype)
        else:
            print("in eat(), norm_method not understood")
            sys.exit(-8)
        self.mps[self.cano] = self.mps[self.cano] / norm
        if(norm <= self.cutoff):
            return 0,error,1
        return np.log(norm),error,1

    def reverse(self):
        if(len(self.mps) == 1):
            return
#        print("reversing")
#        print([i.shape for i in self.mps])
        self.neighbor = self.neighbor[::-1]
        self.mps = [i.transpose([2,1,0]) for i in self.mps[::-1]]
        self.cano = len(self.mps) - 1-self.cano
#        print([i.shape for i in self.mps])
#        print("done")

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
        if(len(self.mps)==0):
            return 0,1
        if(len(self.mps)==1 and self.mps[0].shape[1] == 1):
            z=self.mps[0].squeeze()
#            print("this should be the last tensor",z)
            return np.log(np.abs(z)),np.sign(z)
        print("mps.lognorm(): Computing norm of a MPS is not a good idea in contraction, check it!!!")
        sys.exit(-9)

    def clear(self):
        self.mps = []
        self.neighbor = []
