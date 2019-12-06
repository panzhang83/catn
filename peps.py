"""
Complex valued peps class
"""
import numpy as np
import math
import networkx as nx
from scipy.linalg import sqrtm
import time
import sys


class PEPS:
    """ Tensor network for the graphical model.
    Storage: The data are stored in a dictionary *tensors*, each of which is a Node class.
    """

    def __init__(self,lx,ly,tensors,neighbors,seed=1,mydevice='cpu',maxdim=30,verbose=-1,Dmax=12,chi=32,node_type="raw",cutoff=1.0e-14):
        self.cutoff=cutoff
        print("cutoff=",self.cutoff)
        self.lx=lx
        self.ly=ly
        self.error = 0
        self.Dmax=Dmax
        self.svd_max_dim = 0
        self.chi=chi
        orders = np.array([len(i.shape) for i in tensors])
        print(orders)
        print(np.sum(orders==1),"order 1")
        print(np.sum(orders==2),"order 2")
        print(np.sum(orders==3),"order 3")
        print(np.sum(orders==4),"order 4")
#       lx rows, ly columns 
        assert(lx <= ly) # this needs to be ensured by the circuit generator !!!
        self.tensors = [[] for i in range(self.lx)]

        coord= [] 

        n = 0
        for i in tensors:
            if len(i.shape) == 1:
                #                print(i.dtype)
#                print(i.shape)
#                print(i.reshape([1,1,1,1,2]))
#                print(n%lx)
                self.tensors[n % lx].append(i.reshape([1,1,1,1,2]))
                coord.append([[n%lx,n//lx]] )
                print(n,coord[-1])
                n = n+1
            else:
                break
        print([len(i) for i in self.tensors])
        print(lx,"x",ly,"lattice")
        print("totally",n,"qubits")
        assert(lx*ly == n)
        for idx in range(n,len(neighbors)):
            #            print(idx,neighbors[idx])
            if(len(neighbors[idx])==1):
                i = neighbors[idx][0]
                idx_j_in_i = np.argwhere(neighbors[i]==idx)[0][0]
                xj,yj = coord[i][idx_j_in_i]
#                print(self.tensors[xj][yj].shape)
                self.tensors[xj][yj] = np.einsum("abcde,e->abcd",self.tensors[xj][yj],tensors[idx])
#                print(self.tensors[xj][yj].shape)
#                print("measurement",xj,yj)
            if(len(neighbors[idx])==2):
                i = neighbors[idx][0]
                idx_j_in_i = np.argwhere(neighbors[i]==idx)[0][0]
                xj,yj = coord[i][idx_j_in_i]
                coord.append([[],[xj,yj]])
#                print(coord[-1])
#                print("tensor shape",self.tensors[xj][yj].shape)
                self.tensors[xj][yj] = np.einsum("abcde,ef->abcdf",self.tensors[xj][yj],tensors[idx])
#                print("tensor shape",self.tensors[xj][yj].shape)
#                input("single qubit gate")
                
            if(len(neighbors[idx])==4):

                i = neighbors[idx][0]
                idx_k_in_i = np.argwhere(neighbors[i]==idx)[0][0]
                xk,yk = coord[i][idx_k_in_i]

                j = neighbors[idx][1]
                idx_l_in_j = np.argwhere(neighbors[j]==idx)[0][0]
                xl,yl = coord[j][idx_l_in_j]

                tensork = self.tensors[xk][yk]
                tensorl = self.tensors[xl][yl]
                [d0,d1,d2,d3] = tensors[idx].shape
                matgate = tensors[idx].transpose([0,2,1,3]).reshape([d0*d2,d1*d3])
                [U,s,V] = np.linalg.svd(matgate)
                V = V.T
                s_eff = s[s>self.cutoff]
                s=np.diag(np.sqrt(s_eff))
                myd = len(s)
                U=U[:,:myd]@s
                V=V[:,:myd]@s
                t3l = U.reshape([d0,d2,myd])
                t3r = V.reshape([d1,d3,myd])
                
                if xk == xl and yk == yl-1: # left-right contraction
                    print("tensor shape",self.tensors[xk][yk].shape,"       ",self.tensors[xl][yl].shape)
                    print("left-right contraction",(xk,yk),"+",(xl,yl))
#                    print(tensors[idx].reshape([4,4]))
                    d = self.tensors[xk][yk].shape[-1]
                    assert( d == self.tensors[xl][yl].shape[-1] )
                    tk=self.tensors[xk][yk]
                    dd = tk.shape[2]
                    tl=self.tensors[xl][yl]
                    self.tensors[xk][yk] = np.einsum("abcde,efg->abcgdf",tk,t3l).reshape(tk.shape[0],tk.shape[1],tk.shape[2]*t3l.shape[2],tk.shape[3],t3l.shape[1])
                    self.tensors[xl][yl] = np.einsum("abcde,efg->agbcdf",tl,t3r).reshape(tl.shape[0]*t3r.shape[2],tl.shape[1],tl.shape[2],tl.shape[3],t3r.shape[1])
                    k0,k1,k2,k3,k4=self.tensors[xk][yk].shape
                    l0,l1,l2,l3,l4=self.tensors[xl][yl].shape
                    if(k0*k1*k3*k4<self.svd_max_dim and l0*l1*l3*l4<self.svd_max_dim):
                        matk = tk.transpose([0,1,3,2,4]).reshape(-1,tk.shape[2],tk.shape[4])
                        matl = tl.reshape(tl.shape[0],-1,tl.shape[4])
                        mat = np.einsum("abc,bde,cefg->afdg",matk,matl,tensors[idx]).reshape(matk.shape[0]*tensors[idx].shape[2],matl.shape[1]*tensors[idx].shape[3])
                        print(k0*k1*k3*k4,l0*l1*l3*l4)
                        print("SVD on ",mat.shape)
                        [U,s,V] = np.linalg.svd(mat)
                        V = V.T
                        s_eff = s[s>self.cutoff]
                        if(self.Dmax<0):
                            myd = len(s_eff)
                        else:
                            myd = min(len(s_eff),self.Dmax)
                        self.error = self.error + s_eff[myd:].sum()
                        s_eff=s_eff[:myd]
                        s=np.diag(np.sqrt(s_eff))
                        U=U[:,:myd]
                        V=V[:,:myd]
                        print(dd*2,"->",myd)
                        matk = (U@s).reshape([matk.shape[0],tensors[idx].shape[2],myd])
                        matl = (s@(V.T)).reshape([myd,matl.shape[1],tensors[idx].shape[3]])
                        self.tensors[xk][yk] = matk.reshape(tk.shape[0],tk.shape[1],tk.shape[3],tensors[idx].shape[2],myd).transpose([0,1,4,2,3])
                        self.tensors[xl][yl] = matl.reshape([-1,tl.shape[1],tl.shape[2],tl.shape[3],tensors[idx].shape[3]])
#                    print(np.abs(self.tensors[xk][yk] - ta).sum(),np.abs(self.tensors[xl][yl] - tb).sum())
                    print("tensor shape",self.tensors[xk][yk].shape,"       ",self.tensors[xl][yl].shape)


                elif xk == xl-1 and yk == yl: # top-bottom contraction
                    #                    print("top-bottom contraction",(xk,yk),"+",(xl,yl))
                    print("tensor shape",self.tensors[xk][yk].shape,"       ",self.tensors[xl][yl].shape)
                    print("top-bottom contraction",(xk,yk),"+",(xl,yl))
                    d = self.tensors[xk][yk].shape[-1]
                    assert( d == self.tensors[xl][yl].shape[-1] )
                    tk=self.tensors[xk][yk]
                    tl=self.tensors[xl][yl]
                    self.tensors[xk][yk] = np.einsum("abcde,efg->abgcdf",tk,t3l).reshape(tk.shape[0],tk.shape[1]*t3l.shape[2],tk.shape[2],tk.shape[3],t3l.shape[1])
                    self.tensors[xl][yl] = np.einsum("abcde,efg->abcdgf",tl,t3r).reshape(tl.shape[0],tl.shape[1],tl.shape[2],tl.shape[3]*t3r.shape[2],t3r.shape[1])

                    k0,k1,k2,k3,k4=self.tensors[xk][yk].shape
                    l0,l1,l2,l3,l4=self.tensors[xl][yl].shape
                    if(k0*k2*k3*k4 < self.svd_max_dim and l0*l2*l3*l4 < self.svd_max_dim):
                        dd = tk.shape[1]
                        matk = tk.transpose([0,2,3,1,4]).reshape(-1,tk.shape[1],tk.shape[4])
                        matl = tl.reshape(-1,tl.shape[3],tl.shape[4])
                        mat = np.einsum("abc,dbe,cefg->afdg",matk,matl,tensors[idx]).reshape(matk.shape[0]*tensors[idx].shape[2],matl.shape[0]*tensors[idx].shape[3])
                        print(k0*k2*k3*k4,l0*l2*l3*l4)
                        print("SVD on ",mat.shape)
                        [U,s,V] = np.linalg.svd(mat)
                        V = V.T
                        s_eff = s[s>self.cutoff]
                        if(self.Dmax<0):
                            myd = len(s_eff)
                        else:
                            myd = min(len(s_eff),self.Dmax)
#                    if(len(s)>myd):
                        print(dd*2,"->",myd)
                        self.error = self.error + s_eff[myd:].sum()
                        s_eff=s_eff[:myd]
                        s=np.diag(np.sqrt(s_eff))
                        U=U[:,:myd]
                        V=V[:,:myd]
                        matk = (U@s).reshape([matk.shape[0],tensors[idx].shape[2],myd])
                        matl = (s@(V.T)).reshape([myd,matl.shape[0],tensors[idx].shape[3]])
                        self.tensors[xk][yk] = matk.reshape(tk.shape[0],tk.shape[2],tk.shape[3],tensors[idx].shape[2],myd).transpose([0,4,1,2,3])
                        self.tensors[xl][yl] = matl.reshape([myd,tl.shape[0],tl.shape[1],tl.shape[2],tensors[idx].shape[3]]).transpose([1,2,3,0,4])
                    print("tensor shape",self.tensors[xk][yk].shape,"       ",self.tensors[xl][yl].shape)

                else:
                    print("Something wrong in self.init")
                    sys.exit(-1)

                coord.append([[],[],[xk,yk],[xl,yl]])
#                print(coord[-1])
#                input("two-qubit gate")

        for i in self.tensors:
            for j in i:
                print(list(j.shape))
        bondim = [math.log2(i.shape[2]) for i in self.tensors[0]]
#        print(bondim)
        bondim2 = [math.log2(i.shape[2]) for i in self.tensors[-1]]
#        print(bondim2)

        for i in range(1,self.lx//2):
            for j in range(self.ly):
                bondim[j] = bondim[j] + math.log2(self.tensors[i][j].shape[2])

        for i in range(self.lx-2,self.lx//2-1,-1):
            for j in range(self.ly):
                bondim2[j] = bondim2[j] + math.log2(self.tensors[i][j].shape[2])
        print([(1,self.tensors[self.lx//2-1][0].shape[1],int(2**bondim[0]))]+[(int(2**bondim[i]),self.tensors[self.lx//2-1][i].shape[1],int(2**bondim[i+1])) for i in range(len(bondim)-1)])
        a = [math.log2(self.tensors[self.lx//2-1][0].shape[1])+bondim[0]]+[ bondim[i]+math.log2(self.tensors[self.lx//2-1][i].shape[1])+bondim[i+1] for i in range(len(bondim)-1)]
        b= [math.log2(self.tensors[self.lx//2][0].shape[3])+bondim2[0]]+[bondim2[i]+math.log2(self.tensors[self.lx//2][i].shape[3])+bondim2[i+1] for i in range(len(bondim2)-1)]
#        a=[(math.log2(self.tensors[self.lx//2-1][0].shape[1])+bondim[0])]+[(bondim[i]+math.log2(self.tensors[self.lx//2-1][i].shape[1])+bondim[i+1]) for i in range(len(bondim)-1)]
#        b=[(math.log2(self.tensors[self.lx//2-1][0].shape[1])+bondim[0])]+[(bondim2[i]+math.log2(self.tensors[self.lx//2][i].shape[3])+bondim2[i+1]) for i in range(len(bondim2)-1)]
        print("should be %.3f GB"%( np.power(2,np.array(a)-30+4).sum() + np.power(2,np.array(b)-30+4).sum()))
#        print([(int(2**bondim2[i]),self.tensors[self.lx//2][i].shape[3],int(2**bondim2[i+1])) for i in range(len(bondim2)-1)])
            
        self.topmps = [i.reshape(i.shape[:-1]) for i in self.tensors[0]]
        self.bottommps = [i.reshape([i.shape[0],i.shape[2],i.shape[3]]).transpose([0,2,1]) for i in self.tensors[-1]]
        tot_mem = 0

        print("computing top boundary mps")
        for i in range(1,self.lx//2):
            for j in range(self.ly):
                self.topmps[j] = np.einsum("abc,defb->adecf",self.topmps[j], self.tensors[i][j]).reshape(self.topmps[j].shape[0]*self.tensors[i][j].shape[0],self.tensors[i][j].shape[1],-1)
                mem=(2**(math.log2(np.prod(self.topmps[j].shape))-30+4))
                tot_mem += mem
                print(i,j,"size: %.3g GB"%mem, "shape",self.topmps[j].shape,"tot_mem %.3g GB"%tot_mem)

        print("computing bottom boundary mps")
        for i in range(self.lx-2,self.lx//2-1,-1):
            for j in range(self.ly):
                self.bottommps[j] = np.einsum("abc,dbef->daecf",self.bottommps[j], self.tensors[i][j]).reshape(self.bottommps[j].shape[0]*self.tensors[i][j].shape[0],self.tensors[i][j].shape[2]*self.bottommps[j].shape[2],self.tensors[i][j].shape[3]).transpose([0,2,1])
                mem=(2**(math.log2(np.prod(self.bottommps[j].shape))-30+4))
                tot_mem += mem
                print(i,j,"size: %.3g GB"%mem, "shape",self.topmps[j].shape,"tot_mem %.3gGB"%tot_mem)

    def contraction(self):
#        a = self.tensors[0][0]
#        b = self.tensors[0][1]
#        c = self.tensors[1][0]
#        d = self.tensors[1][1]
#        z = np.einsum("abcd,cefg,hijb,jkle->adgflkih",a,b,c,d)
#        print("psi=",z,"prob=",np.abs(z)**2)
        mat = self.topmps[0].reshape(self.topmps[0].shape[1:]).T @ self.bottommps[0].reshape(self.bottommps[0].shape[1:])
        for i in range(1,len(self.topmps)):
            mat = np.einsum("abc,ad,dbe->ce",self.topmps[i],mat,self.bottommps[i])
        mat = mat.item()
        print("psi=",mat,"prob=",np.abs(mat)**2)
        sys.exit(0)





