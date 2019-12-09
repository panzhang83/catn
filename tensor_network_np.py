"""
Tensor network class
"""
import numpy as np
import math
import networkx as nx
from scipy.linalg import sqrtm
import time
import sys
import mps_node_np
from npsvd import svd,rsvd


class Tensor_Network_np:
    """ Tensor network for the graphical model.
    Storage: The data are stored in a dictionary *tensors*, each of which is a Node class.
    """

    def __init__(self,tensors,labels,seed=1,mydevice='cpu',maxdim=30,verbose=-1,Dmax=12,chi=32,node_type="np",norm_method=1,svdopt=True,swapopt=True,reverse=True,bins = 20,cutoff=1.0e-15,select=1):
        self.norm_method=norm_method
        self.reverse=reverse
        self.sign=1
        self.bins=bins
        self.n = len(tensors)
        self.print_interval = self.n // self.bins
        self.svdopt = svdopt
        self.swapopt = swapopt
        edges = []
        for i in range(len(labels)):
            for j in labels[i]:
                edges.append(sorted([i,j]))
        self.G = nx.MultiGraph()
        self.node_type = node_type
        self.chi=chi
        self.device=mydevice
        self.verbose=verbose
        np.random.seed(seed)
        self.maxdim=maxdim
        self.Dmax=Dmax # maximum bond dimension
        self.cutoff=cutoff
        self.select= select
        self.G.add_nodes_from(np.arange(self.n))
        self.G.add_edges_from(  set([tuple(sorted(a)) for a in edges]))
        self.m = len(self.G.edges)
        self.maxdim_intermediate = -1
        max_degree = max(np.array(self.G.degree)[:,1])
        self.num_isolated=sum((np.array(self.G.degree)[:,1]) == 0)
        print("totally",self.n,"nodes,",self.m,"edges,","maximum degree=",max_degree,", number of isolated nodes=", self.num_isolated)
#        print(edges)
#        print("degree",self.G.degree)
#        print("G.edges",self.G.edges)

        self.tensors = {}.fromkeys(np.arange(self.n))
        for key in self.tensors.keys():
            tensor = tensors[key]
            self.tensors[key] = mps_node_np.MPSNode(tensor, key,list(labels[key]),self.chi,self.cutoff,self.norm_method,self.svdopt,self.swapopt,self.verbose)
            self.tensors[key].left_canonical()

        self.select_edge_init()

    def dim_after_merge(self,i,j):
        nodei = self.tensors[i]
        nodej = self.tensors[j]
        idx_j_in_i=nodei.find_neighbor(j)
        di = nodei.logdim()
        dj = nodej.logdim()
        d=nodei.logdim(idx_j_in_i)
        return round(di+dj-d*2)

    def select_edge_total_dimension(self):
        edge = np.array(list(self.G.edges()))
        minidx=0
        mind=math.inf
        for i,j in edge:
            count = self.dim_after_merge(i,j) 
            if count<mind:
                mind = count
                myi,myj=i,j
        if(mind>self.maxdim_intermediate):
            self.maxdim_intermediate=mind
        if(mind>self.maxdim):
            print("Tring to contract tensor",i, "and tensor",j,"intermediate tensor dimension",mind)
            nodei = self.tensors[i]
            nodej = self.tensors[j]
            print(i,nodei.shape(),nodei.neighbor)
            print(j,nodej.shape(),nodej.neighbor)
            print("The intermediate tensor is larger than maximum dimension")
            self.print_all_tensor_shape()
            sys.exit(1)
        return myi,myj

    def select_edge_min_dim(self):
        count = min([i if len(self.edge_count[i]) > 0 else math.inf for i in self.edge_count.keys()])
        if(count>self.maxdim_intermediate):
            self.maxdim_intermediate = count
        return self.edge_count[count][0]

    def select_edge_min_dim_triangle(self):
        count = min([i if len(self.edge_count[i]) > 0 else math.inf for i in self.edge_count.keys()])
        if(count>self.maxdim_intermediate):
            self.maxdim_intermediate = count
#            print(self.edge_count[count])
        x = np.random.randint(len(self.edge_count[count]))
#        print("count=",count)
#        print(self.edge_count[count])
        triangle_count = []
#        for i,j in self.edge_count[count]:
        for a in range(len(self.edge_count[count])):
            i,j = self.edge_count[count][a]
            neigh1 = self.tensors[i].neighbor
            neigh2 = self.tensors[j].neighbor
            idx_i_in_j=np.argwhere(neigh2==i)[0][0]
            both = 0
            for l in range(len(neigh2)):
                if l != idx_i_in_j:
                    k=neigh2[l]
                    idx_i_in_k = self.tensors[k].find_neighbor(i)
                    if(idx_i_in_k > -1): # i already in k
                        both = both+1
            triangle_count.append(both)
#        print(triangle_count)
        x = np.array(triangle_count).argmax()
#        print(triangle_count[x])
        return self.edge_count[count][x]


    def count_add_nodes(self,nodes):
        edges = []
        for i in nodes:
            edges = edges + [tuple(sorted([i,j])) for j in self.tensors[i].neighbor]
        self.count_add_edges(set(edges))

    def count_add_edges(self,edges):
        """ Notice that two end nodes of each edge should sorted, and edges should be unique """
        for i,j in edges:
            count = self.dim_after_merge(i,j)
            if count in self.edge_count.keys():
                self.edge_count[count].append(sorted([i,j]))
            else:
                self.edge_count[count]=[sorted([i,j])]

    def select_edge_init(self):
        self.edge_count = {}
        self.count_add_edges(set([tuple(sorted(a)) for a in self.G.edges()]))

    def count_remove_nodes(self,nodes):
        for j in nodes:
            for i in self.tensors[j].neighbor:
                count = self.dim_after_merge(i,j)
                if(sorted([i,j]) in self.edge_count[count]):
                    self.edge_count[count].remove(sorted([i,j]))

    def print_all_tensor_shape(self):
        for i,t in self.tensors.items():
            if(t.shape() != []):
                print(i,t.shape(),t.neighbor)

    def find_low_rank_all_edges(self):
        if(self.verbose >= 1):
            print("Finding low rank structures...");
        error=0
        for i in range(self.n):
            if(len(self.tensors[i].neighbor) <=1 ):
                continue
            for idxj in range(len(self.tensors[i].neighbor)):
                j = self.tensors[i].neighbor[idxj]
                if(len(self.tensors[j].mps) <=1 ):
                    continue
                if self.svdopt:
                    error = error + self.cut_bondim_opt(i,idxj)
                else:
                    error = error + self.cut_bondim(i,idxj)
        if(self.verbose >= 1):
            print("done")
        return error

    def contraction(self):
        error = 0
        self.psi = 1
        self.lnZ = np.log(np.array([2]).astype(self.tensors[0].dtype)) * self.num_isolated
        t_select=0
        t_contract=0
        t_svd=0
        while self.G.number_of_edges() > 0:
            t0 =time.time()
            if(self.select == 0):
                i,j = self.select_edge_min_dim()
            elif(self.select == 1):
                i,j = self.select_edge_min_dim_triangle()
            else:
                print("wrong choice for args.select")
                sys.exit(10)
            if(self.tensors[j].order() > self.tensors[i].order()):
                i,j = j,i # this is to ensure that node i has larger degree than node j
            orderi = self.tensors[i].order()
            orderj = self.tensors[j].order()
            logdimi = self.tensors[i].logdim()
            logdimj = self.tensors[j].logdim()

            self.count_remove_nodes([i,j]+list(self.tensors[i].neighbor)+list(self.tensors[j].neighbor)) # take care of the count dictionary first because it depends on shape of tensors

            t_select += time.time()-t0
            neigh1 = self.tensors[i].neighbor
            neigh2 = self.tensors[j].neighbor
            idx_j_in_i=np.argwhere(neigh1==j)[0][0]
            idx_i_in_j=np.argwhere(neigh2==i)[0][0]
            if(self.reverse):
                if(idx_j_in_i < len(self.tensors[i].neighbor)//2):
                    #                    print("idx_j ",idx_j_in_i)
                    self.tensors[i].reverse()
                    neigh1 = self.tensors[i].neighbor
                    idx_j_in_i=np.argwhere(neigh1==j)[0][0]
#                    print("idx_j ",idx_j_in_i)
                if(idx_i_in_j >= len(self.tensors[j].neighbor)//2):
                    #                    print("idx_i ",idx_i_in_j)
                    self.tensors[j].reverse()
                    neigh2 = self.tensors[j].neighbor
                    idx_i_in_j=np.argwhere(neigh2==i)[0][0]
#                    print("idx_i ",idx_i_in_j)

            t1=time.time()
            self.tensors[i].delete_neighbor(j)
            duplicate=[]
            for l in range(len(neigh2)):
                # arrange neighbors
                if l != idx_i_in_j:
                    k=neigh2[l]
                    idx_k_in_i = self.tensors[i].find_neighbor(k)
                    self.tensors[i].add_neighbor(k)# append the new neighbor to the neighbor list
                    self.G.add_edge(i, k)
                    idx_i_in_k = self.tensors[k].find_neighbor(i)
                    idx_j_in_k = self.tensors[k].delete_neighbor(j)
                    self.tensors[k].add_neighbor(i, idx_j_in_k) # add i to k's neighbor list, replacing j
                    if(idx_i_in_k > -1): # i already in k
                        duplicate.append(k)
                        if(self.verbose >= 1):
                            sys.stdout.write("merging %d %d ..."%(k,i));sys.stdout.flush()
                        error = error + self.tensors[k].merge(i,cross=idx_i_in_k > idx_j_in_k)
                        if(self.verbose >= 1):
                            print("done")
            old_shapei = self.tensors[i].shape()
            old_shapej = self.tensors[j].shape()
            if(self.verbose >= 1):
                sys.stdout.write("eating %d %d ..."%(i,j));sys.stdout.flush()
            lognorm,err,psi = self.tensors[i].eat(self.tensors[j],idx_j_in_i,idx_i_in_j)
            if(self.verbose >= 1):
                print("done")
            error = error+err
            self.psi = self.psi*psi
            self.lnZ += lognorm
            
            for k in duplicate:
                if(self.verbose >= 1):
                    sys.stdout.write("merging %d %d ..."%(i,k));sys.stdout.flush()
                self.tensors[i].merge(k,cross=False)
                if(self.verbose >= 1):
                    print("done")
                idx_k_in_i = self.tensors[i].find_neighbor(k)
                if(self.verbose >= 1):
                    #sys.stdout.write("cutting %d %d ..."%(i,k));sys.stdout.flush()
                    print("cutting %d %d ..."%(i,k))
                if self.svdopt:
                    error = error + self.cut_bondim_opt(i,idx_k_in_i)
                else:
                    error = error + self.cut_bondim(i,idx_k_in_i)
#                if(self.verbose >= 1):
#                    print("done")
            self.tensors[j].clear()
            self.G.remove_node(j)
            t_contract += time.time()-t1
            t0=time.time()
            if(self.verbose >= 1):
                sys.stdout.write("compressing...");sys.stdout.flush()
            if self.svdopt:
                self.tensors[i].compress_opt()
            else:
                self.tensors[i].compress()
            if(self.verbose >= 1):
               print("done")
            """
            #The code below tries to find low-rank structures after compression. However in practice it can not find any low-rank structures.
            if(self.verbose >= 1):
                #sys.stdout.write("check low-rank again...");sys.stdout.flush()
                print("check low-rank again...")
            for idxj in range(len(self.tensors[i].neighbor)):
                j = self.tensors[i].neighbor[idxj]
                if(len(self.tensors[j].mps) <=1 ):
                    continue
                if self.svdopt:
                    error = error + self.cut_bondim_opt(i,idxj)
                else:
                    error = error + self.cut_bondim(i,idxj)

            if(self.verbose >= 1):
                print("done")
            """


            edges=np.array(list(self.G.edges))
            m_left=0
            if(len(edges)>0):
                edges = edges[:,:2]
                m_left=len(np.unique(edges,axis=0))

#            if(m_left>2 and self.Dmax>0):
#                error = error + self.low_rank_approx_site(i)

            t_svd += time.time()-t0
            n_left=self.num_tensor_remain()
            duplicate_str = " ".join([str(ii) for ii in duplicate])
            if(self.verbose < 1):
                if(m_left< 100 or (n_left % self.print_interval == 0)):
                    print("%d/%d"%(m_left,self.m),"%d/%d"%(n_left,len(self.tensors)),"err=%.3e"%np.abs(error),"lnZ=%.3e"%np.real(self.lnZ),"%d, %d -> %d"%(orderi,orderj,self.tensors[i].order()), "\t%.2f"%(time.time()-t1),"Sec.")
            else:
                print("%d/%d"%(m_left,self.m),"%d/%d"%(n_left,len(self.tensors)),"(%d,%d)"%(i,j),"err=%.3e"%np.abs(error),"lnZ=%.3e"%np.real(self.lnZ),"%d, %d -> %d   [%s],"%(orderi,orderj,self.tensors[i].order(),duplicate_str),"%.1f %.1f %.1f"%(logdimi,logdimj,self.tensors[i].logdim()), "\t%.2f"%(time.time()-t1),"Sec.")
                #print([str(i) for i in self.tensors[i].logdim()])
#                print(self.tensors[i].logdim())
            self.count_add_nodes([i]+list(self.tensors[i].neighbor))
        lognorm,self.sign = self.lognorm()
        self.lnZ = self.lnZ + lognorm
        return self.lnZ,error,self.psi

    def lognorm(self):
        lognorm = np.array(0).astype(self.tensors[0].dtype)
        for i in self.tensors.keys():
            lognormi,sign = self.tensors[i].lognorm()
            lognorm = lognorm + lognormi
        return lognorm,sign

    def low_rank_approx_site(self,i):
        """ Try to do low-dimensional approximations to large bond fo site i"""

        error = 0
        if(self.tensors[i].shape() == []):
            return 0
        t=self.tensors[i]
        try:
            if(t.order()==0):
               return 0
        except:
            print("error in low_rank_approximate(",i,")","tensor")
            print(t.tensor)
            sys.exit(2)

        while max(self.tensors[i].shape()) > self.Dmax:
            if self.svdopt:
                error = error + self.cut_bondim_opt(i,np.array(self.tensors[i].shape()).argmax())
            else:
                error = error + self.cut_bondim(i,np.array(self.tensors[i].shape()).argmax())
        return error

    def cut_bondim(self,i,idx_j_in_i):
        error = 0

        j=self.tensors[i].neighbor[idx_j_in_i]
        idx_i_in_j = self.tensors[j].find_neighbor(i)
        if(self.verbose >=1):
            sys.stdout.write("  %s,%s --->"%(str(list(self.tensors[i].mps[idx_j_in_i].shape)),str(list(self.tensors[j].mps[idx_i_in_j].shape))));
            sys.stdout.flush()
        da_l = self.tensors[i].mps[idx_j_in_i].shape[0]
        da_r = self.tensors[i].mps[idx_j_in_i].shape[2]
        d = self.tensors[i].mps[idx_j_in_i].shape[1]

        db_l = self.tensors[j].mps[idx_i_in_j].shape[0]
        db_r = self.tensors[j].mps[idx_i_in_j].shape[2]

        mati = self.tensors[i].mps[idx_j_in_i].transpose([0,2,1]).reshape(-1,d)

        matj = self.tensors[j].mps[idx_i_in_j].transpose([0,2,1]).reshape(-1,d)
        merged_matrix = mati@matj.T
        try:
            [U,s,V] = svd(merged_matrix)
        except:
            print("SVD failed: shape of merged_matrix",merged_matrix.shape)
            sys.exit(-1)
        s_eff = s[s>self.cutoff]
        if(len(s_eff) == 0):
            s_eff = s[:1]
        error += s[len(s_err):].sum()
        myd = min(len(s_eff),self.Dmax)
        if(myd == 0):
            print("Warning: encountered ZERO matrix in cut_bondim()")
            myd = 1
            mati=(U[:,0]*s[0])[:,None]
            matj = ((s[0]*V[:,0].T).T)[:,None]
        else:
            error = error + s_eff[myd:].sum()
            s_eff=s_eff[:myd]
            s=np.diag(np.sqrt(s_eff))
            U=U[:,:myd]
            V=V[:,:myd]
            mati=U@s
            matj = (s@V.T).T
        mati = mati.reshape(da_l,da_r,mati.shape[1]).transpose([0,2,1])
        self.tensors[i].mps[idx_j_in_i] = mati
        matj = matj.reshape(db_l,db_r,matj.shape[1]).transpose([0,2,1])
        self.tensors[j].mps[idx_i_in_j] = matj

#        print(list(self.tensors[i].mps[idx_j_in_i].shape),list(self.tensors[j].mps[idx_i_in_j].shape));
        return error

    def cut_bondim_opt2(self,i,idx_j_in_i):
        error = 0

        j=self.tensors[i].neighbor[idx_j_in_i]
        idx_i_in_j = self.tensors[j].find_neighbor(i)
        self.tensors[i].cano_to(idx_j_in_i)
        self.tensors[j].cano_to(idx_i_in_j)
#        print("cano_i",self.tensors[i].cano,idx_j_in_i)
#        print("cano_j",self.tensors[j].cano,idx_i_in_j)
        if(self.verbose >=1):
            sys.stdout.write("  %s,%s --->"%(str(list(self.tensors[i].mps[idx_j_in_i].shape)),str(list(self.tensors[j].mps[idx_i_in_j].shape))));
            sys.stdout.flush()
        da_l = self.tensors[i].mps[idx_j_in_i].shape[0]
        da_r = self.tensors[i].mps[idx_j_in_i].shape[2]
        d = self.tensors[i].mps[idx_j_in_i].shape[1]

        db_l = self.tensors[j].mps[idx_i_in_j].shape[0]
        db_r = self.tensors[j].mps[idx_i_in_j].shape[2]

        mati = self.tensors[i].mps[idx_j_in_i].transpose([0,2,1]).reshape(da_l*da_r,d)

        matj = self.tensors[j].mps[idx_i_in_j].transpose([0,2,1]).reshape(db_l*db_r,d)

        flag = False
        #if(mati.shape[0]*matj.shape[0] < mati.shape[1]*matj.shape[1]):
        if(1==2):
            merged_matrix = mati@matj.T
        else:
            flag=True
            qi,ri = np.linalg.qr(mati)
            qj,rj = np.linalg.qr(matj)
            merged_matrix = ri@rj.T

        [U,s,V] = svd(merged_matrix)
        s_eff = s[s>self.cutoff]
        if(len(s_eff) == 0):
            s_eff = s[:1]
        error = error + s[len(s_err):].sum()
        myd = min(len(s_eff),self.Dmax)
        
        if(myd == 0):
            print("Warning: encountered ZERO matrix in cut_bondim()")
            myd = 1
            mati=(U[:,0]*s[0])[:,None]
            matj = ((s[0]*V[:,0].T).T)[:,None]
        else:
            error = error + s_eff[myd:].sum()
            s_eff=s_eff[:myd]
            s=np.diag(np.sqrt(s_eff))
            U=U[:,:myd]
            V=V[:,:myd]
            mati = U@s
            matj = (s@V.T).T
        if flag:
            mati = qi @ mati
            matj = qj @ matj

        mati = mati.reshape(da_l,da_r,mati.shape[1]).transpose([0,2,1])
        self.tensors[i].mps[idx_j_in_i] = mati
        matj = matj.reshape(db_l,db_r,matj.shape[1]).transpose([0,2,1])
        self.tensors[j].mps[idx_i_in_j] = matj

        if(self.verbose >=1):
            print(list(self.tensors[i].mps[idx_j_in_i].shape),list(self.tensors[j].mps[idx_i_in_j].shape));
        return error

    def cut_bondim_opt(self,i,idx_j_in_i):
        error = 0

        j=self.tensors[i].neighbor[idx_j_in_i]
        idx_i_in_j = self.tensors[j].find_neighbor(i)
        self.tensors[i].cano_to(idx_j_in_i)
        self.tensors[j].cano_to(idx_i_in_j)
#        print("cano_i",self.tensors[i].cano,idx_j_in_i)
#        print("cano_j",self.tensors[j].cano,idx_i_in_j)
        Dold = self.tensors[i].mps[idx_j_in_i].shape[1]
        if(self.verbose >=1):
            sys.stdout.write("  %s,%s ---> "%(str(list(self.tensors[i].mps[idx_j_in_i].shape)),str(list(self.tensors[j].mps[idx_i_in_j].shape))));
            sys.stdout.flush()
        da_l = self.tensors[i].mps[idx_j_in_i].shape[0]
        da_r = self.tensors[i].mps[idx_j_in_i].shape[2]
        d = self.tensors[i].mps[idx_j_in_i].shape[1]

        db_l = self.tensors[j].mps[idx_i_in_j].shape[0]
        db_r = self.tensors[j].mps[idx_i_in_j].shape[2]

        mati = self.tensors[i].mps[idx_j_in_i].transpose([0,2,1]).reshape(da_l*da_r,d)

        matj = self.tensors[j].mps[idx_i_in_j].transpose([0,2,1]).reshape(db_l*db_r,d)

        flag = False
        #if(mati.shape[0]*matj.shape[0] < mati.shape[1]*matj.shape[1]):
#        if(1==2):
#            merged_matrix = mati@matj.T
#        else:
#            flag=True
#            qi,ri = np.linalg.qr(mati)
#            qj,rj = np.linalg.qr(matj)
#            merged_matrix = ri@rj.T
#
        flag_left = False
        if(mati.shape[0] > mati.shape[1]):
            qi,ri = np.linalg.qr(mati)
            flag_left = True
        else:
            ri = mati

        flag_right = False
        if(matj.shape[0] > matj.shape[1]):
            qj,rj = np.linalg.qr(matj)
            flag_right = True
        else:
            rj = matj

        
        merged_matrix = ri@rj.T


        [U,s,V] = svd(merged_matrix)
#        s_str = str(["%.3f"%t for t in s])
        s_bak = s
        s_eff = s[s>self.cutoff]
        if(len(s_eff) == 0):
            s_eff = s[:1]
        error = error + s[len(s_eff):].sum()
        myd = min(len(s_eff),self.Dmax)
        
        if(myd == 0):
            print("Warning: encountered ZERO matrix in cut_bondim()")
            myd = 1
            mati=(U[:,0]*s[0])[:,None]
            matj = ((s[0]*V[:,0].T).T)[:,None]
        else:
            error = error + s[myd:].sum()
            s_eff=s_eff[:myd]
            s=np.diag(np.sqrt(s_eff))
            U=U[:,:myd]
            V=V[:,:myd]
            mati = U@s
            matj = (s@V.T).T
#        if flag:
#            mati = qi @ mati
#            matj = qj @ matj
        if flag_left:
            mati = qi @ mati
        if flag_right:
            matj = qj @ matj

        mati = mati.reshape(da_l,da_r,mati.shape[1]).transpose([0,2,1])
        self.tensors[i].mps[idx_j_in_i] = mati
        matj = matj.reshape(db_l,db_r,matj.shape[1]).transpose([0,2,1])
        self.tensors[j].mps[idx_i_in_j] = matj
        if(self.verbose >=1):
            if(self.tensors[i].mps[idx_j_in_i].shape[1] < Dold):
                sys.stdout.write(str([list(self.tensors[i].mps[idx_j_in_i].shape),list(self.tensors[j].mps[idx_i_in_j].shape)]));
#                sys.stdout.write(" %s"%s_str)
#                print(s_bak)
                print(" ")
            else:
                print(" ")
        return error

    def num_tensor_remain(self):
        return np.sum(np.array([1 if len(self.tensors[i].mps)>0 else 0 for i in self.tensors.keys()]))

    def resources_remain(self):
        #return 2**(np.sum(np.array([self.tensors[i].logdim() for i in self.tensors.keys()]))-30)
        return (np.sum(np.array([self.tensors[i].logdim() for i in self.tensors.keys()]))-30)

