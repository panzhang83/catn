"""
Tensor network class
"""
import torch
import numpy as np
import math
import networkx as nx
from scipy.linalg import sqrtm
import time
import sys
import raw_node
import mps_node
sys.path.append('..')
from torchsvd import SVD
from args import args

svd = SVD.apply

class Tensor_Network:
    """ Tensor network for the graphical model.
    Storage: The data are stored in a dictionary *tensors*, each of which is a Node class.
    """

  

    def __init__(self, n, valtype,edges, weights, fields, beta, seed=1, mydevice='cpu', maxdim=30, verbose=-1, Dmax=1024,
                 chi=32, node_type="raw"):
        self.n = n
        self.G = nx.MultiGraph()
        self.node_type = node_type
        self.chi = chi
        self.device = mydevice
        self.verbose = verbose
        np.random.seed(seed)
        self.maxdim = maxdim
        self.Dmax = Dmax  # maximum bond dimension
        self.cutoff = 1.0e-15
        self.G.add_nodes_from(np.arange(self.n))
        self.G.add_edges_from(edges)
        self.m = len(self.G.edges)
        self.maxdim_intermediate = -1
        max_degree = max(np.array(self.G.degree)[:, 1])
        self.num_isolated = sum((np.array(self.G.degree)[:, 1]) == 0)
        print("maximum degree=", max_degree, ", number of isolated nodes=", self.num_isolated)
        '''
        Is = [torch.tensor([])]
        for i in range(1, max_degree + 2):
            tensor = torch.zeros(2 ** i, dtype=torch.float64, device=mydevice)
            tensor[0] = 1
            tensor[-1] = 1
            Is.append(tensor.reshape([2] * i))
        '''
        # self.tensors = {}.fromkeys(np.arange(self.n))
        self.tensors = []
        for key in range(self.n):
            
            if self.node_type == "raw":
                #tensor = Is[self.G.degree(key) + 1]
                tensor=self.get_identensor(self.G.degree(key)+1,valtype[key],mydevice)
                tensor = tensor.reshape(-1, tensor.shape[-1])
                if len(fields[key].shape) ==0:
                   hi = torch.exp(fields[key] * torch.tensor([1, -1], dtype=torch.float64, device=mydevice)).reshape(
                    2, 1)
                elif len(fields[key].shape) ==1:
                   hi=fields[key].clone().detach().reshape(valtype[key],1)
                   #hi=torch.tensor(fields[key],dtype=torch.float64, device=mydevice).reshape(valtype[key],1)
                   #print(hi)
                else:
                   print("field data structure not  understood")	
                   sys.exit(-8)				   
                tensor = (tensor @ hi).reshape([valtype[key]] * self.G.degree(key))
                self.tensors.append(raw_node.Node(tensor, key, []))
            elif self.node_type == "mps" :
                self.tensors.append(mps_node.MPSNode(torch.tensor([], dtype=torch.float64, device=mydevice), key, [],
                                                     self.chi))
                self.tensors[key].mps = []
            else:
                print("Wrong node_type")
                sys.exit(-5)

        for edge in range(len(edges)):
            i, j = edges[edge]
            if len(weights[edge].shape) == 0:  # a number, simply weight
                B = torch.exp(weights[edge] * beta * torch.tensor([[1, -1], [-1, 1]],
                                                                  dtype=torch.float64, device=self.device))
            elif len(weights[edge].shape) == 2 :  # factor matrix
                # B = weights[edge] * torch.tensor([1], dtype=torch.float64, device=self.device)
                #B = weights[edge]
                B=weights[edge].clone().detach()
            else:
                print("weight data structure not understood")
                sys.exit(-7)

            U, s, V = svd(B)
            s = torch.diag(torch.sqrt(s))
            Q = B
            R = torch.eye(B.shape[1],dtype=torch.float64,device=self.device)
            
            #            Q=torch.tensor(sqrtm([[np.exp(beta), np.exp(-beta)], [np.exp(-beta), np.exp(beta)]]),dtype=torch.float64,device=self.device)
            #            R=Q

            if self.node_type == "raw":
                nodei = self.tensors[i]
                idx = len(nodei.neighbor)
                shapei = [a for a in nodei.tensor.shape]
                shapei[-1]=Q.shape[1]
                #print(nodei.tensor)
                #print(Q)
                nodei.tensor = (nodei.tensor.reshape(-1, Q.shape[0]) @ Q).reshape(shapei)
                if idx < nodei.order() - 1:
                    nodei.tensor = nodei.tensor.permute([a for a in range(1, nodei.order())] + [0])
                nodei.neighbor.append(j)

                nodej = self.tensors[j]
                shapej = [a for a in nodej.tensor.shape]
                
                shapej[-1]=R.shape[1]
                seq = [a for a in range(nodej.order())]
                idx = len(nodej.neighbor)
                x=nodej.tensor.reshape(-1, R.shape[0]) @ R
                
                nodej.tensor = (x).reshape(shapej)
                if idx < nodej.order() - 1:
                    nodej.tensor = nodej.tensor.permute([a for a in range(1, nodej.order())] + [0])
                nodej.neighbor.append(i)

            elif self.node_type == "mps":
                nodei = self.tensors[i]
                if len(fields[i].shape) ==0:
                   fieldi = torch.diag(
                       torch.exp(fields[i] * torch.tensor([1, -1], dtype=torch.float64, device=mydevice)))
                elif len(fields[i].shape) ==1:
                   fieldi = torch.diag(fields[i])
                   #fieldi = torch.diag(torch.tensor(fields[i],dtype=torch.float64, device=mydevice))
                else:
                   print("not understood field structure")
                if self.G.degree(i) == 1:
                    mat = (fieldi @ Q)
                    mat = mat.sum(0)
                    nodei.mps.append(mat.reshape([1, Q.shape[1], 1]))
                else:
                    if len(nodei.neighbor) == 0:
                        # Q.shape[0] is the internal dimension chi, the rank, that is, the dimension of the identity matrix.
                        # Q.shape[1] is the physical dimesion d, could be arbitrary
                        mat = (fieldi @ Q).t()  # notice that the physical dimension could have lower or higher dimension, but the inner dimension should be 2
                        nodei.mps.append(mat.reshape([1, Q.shape[1], Q.shape[0]]))
                    elif len(nodei.neighbor) == self.G.degree(i) - 1:
                        mat = Q
                        nodei.mps.append(mat.reshape([Q.shape[0], Q.shape[1], 1]))
                    else:
                       
                        t3 = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[0], dtype=torch.float64, device=self.device)
                        for m in range(Q.shape[1]):
                            t3[:,m,:]=torch.diag(Q[:,m])
						
                        #mat0 = torch.diag(Q[:, 0])  # chi x chi
                        #mat1 = torch.diag(Q[:, 1])  # chi x chi
                        #t3[:, 0, :] = mat0
                        #t3[:, 1, :] = mat1
                        nodei.mps.append(t3)
 
                nodei.neighbor.append(j)
                
                

                nodej = self.tensors[j]
                if len(fields[j].shape) ==0:
                   fieldj = torch.diag(
                       torch.exp(fields[j] * torch.tensor([1, -1], dtype=torch.float64, device=mydevice)))
                elif len(fields[j].shape) ==1:
                   fieldj = torch.diag(fields[j])
                   #fieldj = torch.diag(torch.tensor(fields[j],dtype=torch.float64, device=mydevice))
                else:
                   print("not understood field structure")
                if self.G.degree(j) == 1:
                    mat = (fieldj @ R)
                    mat = mat.sum(0)
                    nodej.mps.append(mat.reshape([1, R.shape[1], 1]))
                else:
                    if len(nodej.neighbor) == 0:
                        # Q.shape[0] is the internal dimension chi, the rank, that is, the dimension of the identity matrix.
                        # Q.shape[1] is the physical dimesion d, could be arbitrary
                        mat = (fieldj @ R).t()  # notice that the physical dimension could have lower or higher dimension, but the inner dimension should be 2
                        nodej.mps.append(mat.reshape([1, R.shape[1], R.shape[0]]))
                    elif len(nodej.neighbor) == self.G.degree(j) - 1:
                        mat = R
                        nodej.mps.append(mat.reshape([R.shape[0], R.shape[1], 1]))
                    else:
                        t3 = torch.zeros(R.shape[0], R.shape[1], R.shape[0], dtype=torch.float64, device=self.device)
                        for m in range(R.shape[1]):
                            t3[:,m,:]=torch.diag(R[:,m])
						
                        #mat0 = torch.diag(Q[:, 0])  # chi x chi
                        #mat1 = torch.diag(Q[:, 1])  # chi x chi
                        #t3[:, 0, :] = mat0
                        #t3[:, 1, :] = mat1
                        nodej.mps.append(t3)
                nodej.neighbor.append(i)
        self.select_edge_init()
    def get_identensor(self,order,multi,mydevice):
        #print(multi)
        #print(order)
        tensor = torch.zeros(multi ** order, dtype=torch.float64, device=mydevice)
        if multi==1:
           distance=0
        else:
           distance=int((multi**order-1)/(multi-1))
        for  i in range(multi):
             tensor[i*distance] = 1
        
        tensor=tensor.reshape([multi] * order)
        return tensor
    def dim_after_merge(self, i, j):
        nodei = self.tensors[i]
        nodej = self.tensors[j]
        idx_j_in_i = nodei.find_neighbor(j)
        di = nodei.logdim()
        dj = nodej.logdim()
        d = nodei.logdim(idx_j_in_i)
        return round(di + dj - d * 2)

    def select_edge(self):
        count = min([i if len(self.edge_count[i]) > 0 else math.inf for i in self.edge_count.keys()])
        if count > self.maxdim_intermediate:
            self.maxdim_intermediate = count
        if count > self.maxdim and self.node_type == "raw":
            i, j = self.edge_count[count][0]
            print("Tring to contract tensor", i, "and tensor", j, "intermediate tensor dimension", count)
            nodei = self.tensors[i]
            nodej = self.tensors[j]
            print(i, nodei.shape(), nodei.neighbor)
            print(j, nodej.shape(), nodej.neighbor)
            print("The intermediate tensor is larger than maximum dimension")
            self.print_all_tensor_shape()
            sys.exit(1)

        return self.edge_count[count][0]
    def select_edge_total_dimension(self):
        edge = np.array(list(self.G.edges()))
        minidx=0
        mind=10000000
        for a in edge:
            i,j=a
            nodei = self.tensors[i]
            nodej = self.tensors[j]
            neigh1 = nodei.neighbor
            neigh2 = nodej.neighbor
            idxj=np.argwhere(neigh1==j)[0][0]
            idxi=np.argwhere(neigh2==i)[0][0]
            di = nodei.logdim()
            dj = nodej.logdim()
            d=nodei.logdim(idxj)
            count = di+dj-d*2

            if count<mind:
                mind = count
                myi,myj=a
        if(mind>self.maxdim_intermediate):
            self.maxdim_intermediate=mind
        if(mind>self.maxdim):
            print("Tring to contract tensor",i, "and tensor",j)
            print(i,nodei.tensor.shape,nodei.neighbor)
            print(j,nodej.tensor.shape,nodej.neighbor)
            print("The intermediate tensor is larger than maxmum dimension")
            self.print_all_tensor_shape()
            sys.exit(1)
        return myi,myj
    def count_add_nodes(self, nodes):
        edges = []
        for i in nodes:
            edges = edges + [tuple(sorted([i, j])) for j in self.tensors[i].neighbor]
        self.count_add_edges(set(edges))

    def count_add_edges(self, edges):
        """ Notice that two end nodes of each edge should sorted, and edges should be unique """
        for i, j in edges:
            count = self.dim_after_merge(i, j)
            if count in self.edge_count.keys():
                self.edge_count[count].append(sorted([i, j]))
            else:
                self.edge_count[count] = [sorted([i, j])]

    def select_edge_init(self):
        self.edge_count = {}
        self.count_add_edges(set([tuple(sorted(a)) for a in self.G.edges()]))

    def count_remove_nodes(self, nodes):
        for j in nodes:
            for i in self.tensors[j].neighbor:
                count = self.dim_after_merge(i, j)
                if sorted([i, j]) in self.edge_count[count]:
                    self.edge_count[count].remove(sorted([i, j]))

    def print_all_tensor_shape(self):
        for i in range(len(self.tensors)):
            if len(self.tensors[i].tensor) != 0:
                print(i, self.tensors[i].tensor.shape, self.tensors[i].neighbor)
    def select_edge_sequentially(self):
        edge = np.array(list(self.G.edges()))
        sum_edge=edge[:,0]+edge[:,1]
        #print(sum_edge)
        index=np.argmin(sum_edge)
        # i, j = pool[np.random.choice(np.where(count == count.max())[0])]
        #print(edge)
        #print(index)
        i, j = edge[index]
        return i, j
    def contraction(self):
        # self.lnZ = math.log(2) * self.num_isolated
        self.lnZ = torch.log(torch.tensor([2], dtype=torch.float64, device=self.device)) * self.num_isolated
        t_select = 0
        t_contract = 0
        t_svd = 0
        while self.G.number_of_edges() > 0:
            
            t0 = time.time()
            #i,j=self.select_edge_total_dimension()
            if args.corder:
               i,j=self.select_edge_sequentially()
            i, j = self.select_edge()
            if self.tensors[j].order() > self.tensors[i].order():
                i, j = j, i  # this is to ensure that node i has larger degree than node j
#            print(i,j)
            self.count_remove_nodes([i, j] + list(self.tensors[i].neighbor) + list(self.tensors[j].neighbor))
            t_select += time.time() - t0
            neigh1 = self.tensors[i].neighbor
            neigh2 = self.tensors[j].neighbor
            #print(neigh1)
            #print(neigh2)
            #for l in range(len(neigh1)):
                #if neigh1[l]==j:
                   #idx_j_in_i=l 
            #print(idx_j_in_i)
            idx_j_in_i = np.argwhere(np.array(neigh1) == j)[0][0]
            idx_i_in_j = np.argwhere(np.array(neigh2) == i)[0][0]

            t1 = time.time()
            self.tensors[i].delete_neighbor(j)
            duplicate = []
            for l in range(len(neigh2)):
                # arrange neighbors
                if l != idx_i_in_j:
                    k = neigh2[l]
                    #idx_k_in_i = self.tensors[i].find_neighbor(k)
                    self.tensors[i].add_neighbor(k)  # append the new neighbor to the neighbor list
                    self.G.add_edge(i, k)
                    #a=self.tensors[k]
                    idx_i_in_k = self.tensors[k].find_neighbor(i)
                    idx_j_in_k = self.tensors[k].delete_neighbor(j)
                    self.tensors[k].add_neighbor(i, idx_j_in_k)  # add i to k's neighbor list, replaceing j
                    if idx_i_in_k > -1:  # i already in k
                        duplicate.append(k)
                        self.tensors[k].merge(i, cross=idx_i_in_k > idx_j_in_k)
            old_shapei = self.tensors[i].shape()
            old_shapej = self.tensors[j].shape()
            lognorm = self.tensors[i].eat(self.tensors[j], idx_j_in_i, idx_i_in_j)
            self.lnZ += lognorm
            for k in duplicate:
                self.tensors[i].merge(k, cross=False)
                idx_k_in_i = self.tensors[i].find_neighbor(k)
                if self.node_type == "mps" and self.tensors[i].mps[idx_k_in_i].shape[1] > self.Dmax and self.Dmax>0:
                    self.cut_bondim(i, idx_k_in_i)
            self.tensors[j].clear()
            
            self.G.remove_node(j)
            t_contract += time.time() - t1
            t0 = time.time()
           
            edges = np.array(list(self.G.edges))
            m_left = 0
            if len(edges) > 0:
                edges = edges[:, :2]
                m_left = len(np.unique(edges, axis=0))

            if m_left > 2 and self.Dmax > 0:
                self.low_rank_approx_site(i)

            t_svd += time.time() - t0
            if self.verbose > 3:
                print(m_left, "/", self.m, list(old_shapei), "idx", idx_j_in_i, list(old_shapej), "\n\t\t\t\t---->",
                      list(self.tensors[i].shape()))
            if self.verbose > 2:
                print(m_left, "/", self.m, "(%d,%d)" % (i, j), ")", "select %.2f" % t_select,
                      "contact %.2f" % t_contract, "rank_appr. %.2f" % t_svd, self.tensors[i].shape(),
                      "%.2f" % (time.time() - t1), "Sec.")
            else:
                print(m_left, "/", self.m, "(%d,%d)" % (i, j), "\t--->",
                      "%d",self.tensors[i].logdim(), "\t%.2f" % (time.time() - t1), "Sec.")
            self.count_add_nodes([i] + list(self.tensors[i].neighbor))
        lognorm = self.lognorm()
        self.lnZ = self.lnZ + lognorm
        return self.lnZ

    def lognorm(self):
        lognorm = torch.tensor(0, dtype=torch.float64, device=self.device)
        for i in range(self.n):
            lognormi = self.tensors[i].lognorm()
            lognorm = lognorm + lognormi
        return lognorm

    def low_rank_approx(self):
        for i in range(self.n):
            self.low_rank_approx_site(i)

    def low_rank_approx_site(self, i):
        """ Try to do low-dimensional approximations to large bond fo site i"""
        if not self.tensors[i].shape():
            return
        t = self.tensors[i]
        try:
            if t.order() == 0:
                return
        except:
            print("error in low_rank_approximate(", i, ")", "tensor")
            #print(t.tensor)
            sys.exit(2)

        while max(self.tensors[i].shape()) > self.Dmax:
            self.cut_bondim(i, np.array(self.tensors[i].shape()).argmax())

    def cut_bondim_old(self, i, idx_j_in_i):
        j = self.tensors[i].neighbor[idx_j_in_i]
        idx_i_in_j = self.tensors[j].find_neighbor(i)
        if self.node_type == "raw":
            mati = self.tensors[i].unfolding(idx_j_in_i)
            matj = self.tensors[j].unfolding(idx_i_in_j)
            merged_matrix = mati @ matj.t()
        else:

            self.tensors[i].move2tail(idx_j_in_i)
            self.tensors[i].move2tail_neighbor(idx_j_in_i)

            self.tensors[j].move2tail(idx_i_in_j)
            self.tensors[j].move2tail_neighbor(idx_i_in_j)

            mati = self.tensors[i].mps[-1].reshape(self.tensors[i].mps[-1].shape[:2])
            matj = self.tensors[j].mps[-1].reshape(self.tensors[j].mps[-1].shape[:2])
            merged_matrix = mati @ matj.t()
        try:
            [U, s, V] = svd(merged_matrix)
        except:
            print("SVD failed: shape of merged_matrix", merged_matrix.shape)
            sys.exit(-1)
        s_eff = s[s > self.cutoff]
        myd = min(len(s_eff), self.Dmax)
        if myd == 0:
            print("Warning: encountered ZERO matrix in cut_bondim()")
            myd = 1
            mati = (U[:, 0] * s[0])[:, None]
            matj = ((s[0] * V[:, 0].t()).t())[:, None]
        else:
            s_eff = s_eff[:myd]
            s = torch.diag(torch.sqrt(s_eff))
            U = U[:, :myd]
            V = V[:, :myd]
            mati = U @ s
            matj = (s @ V.t()).t()
        if self.node_type == "raw":
            self.tensors[i].restore_from_matrix(mati, idx_j_in_i)
            self.tensors[j].restore_from_matrix(matj, idx_i_in_j)
        else:
            self.tensors[i].mps[-1] = mati.reshape(mati.shape[0], mati.shape[1], 1)
            self.tensors[j].mps[-1] = matj.reshape(matj.shape[0], matj.shape[1], 1)
    def cut_bondim(self,i,idx_j_in_i):
        error = 0
        j=self.tensors[i].neighbor[idx_j_in_i]
        idx_i_in_j = self.tensors[j].find_neighbor(i)
        sys.stdout.flush()
#        print("cutting bond",i,j,"idx",idx_j_in_i,idx_i_in_j)
        if(self.node_type == "raw"):
            mati = self.tensors[i].unfolding(idx_j_in_i)
            matj = self.tensors[j].unfolding(idx_i_in_j)
            merged_matrix = mati@matj.t()
        else:
            da_l = self.tensors[i].mps[idx_j_in_i].shape[0]
            da_r = self.tensors[i].mps[idx_j_in_i].shape[2]
            d = self.tensors[i].mps[idx_j_in_i].shape[1]

            db_l = self.tensors[j].mps[idx_i_in_j].shape[0]
            db_r = self.tensors[j].mps[idx_i_in_j].shape[2]

            mati = self.tensors[i].mps[idx_j_in_i].permute([0,2,1]).reshape(-1,d)

            matj = self.tensors[j].mps[idx_i_in_j].permute([0,2,1]).reshape(-1,d)
            merged_matrix = mati@matj.t()
            merged_matrxi = torch.einsum("ijk,ajb->ikab",self.tensors[i].mps[idx_j_in_i],self.tensors[j].mps[idx_i_in_j]).reshape(da_l*da_r,db_l*db_r)
        try:
            [U,s,V] = svd(merged_matrix)
        except:
            print("SVD failed: shape of merged_matrix",merged_matrix.shape)
            sys.exit(-1)
        s_eff = s[s>self.cutoff]
        myd = min(len(s_eff),self.Dmax)
        if(myd == 0):
            print("Warning: encountered ZERO matrix in cut_bondim()")
            myd = 1
            mati=(U[:,0]*s[0])[:,None]
            matj = ((s[0]*V[:,0].t()).t())[:,None]
        else:
            error = error + s[myd:].sum()
            s_eff=s_eff[:myd]
            s=torch.diag(torch.sqrt(s_eff))
            U=U[:,:myd]
            V=V[:,:myd]
            mati=U@s
            matj = (s@V.t()).t()
        if(self.node_type == "raw"):
            self.tensors[i].restore_from_matrix(mati,idx_j_in_i)
            self.tensors[j].restore_from_matrix(matj,idx_i_in_j)
        else:
            mati = mati.reshape(da_l,da_r,mati.shape[1]).permute([0,2,1])
            self.tensors[i].mps[idx_j_in_i] = mati
            matj = matj.reshape(db_l,db_r,matj.shape[1]).permute([0,2,1])
            self.tensors[j].mps[idx_i_in_j] = matj

        #print(list(self.tensors[i].mps[idx_j_in_i].shape),list(self.tensors[j].mps[idx_i_in_j].shape));
        return error

