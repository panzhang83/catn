"""
Computing LnZ of a graphical model using tensor networks
"""
import torch
import numpy as np
import math
import networkx as nx
import string
import time
import sys
from tensor_network import Tensor_Network
from args import args
from bp_mf import MeanField


def readgraph(D, graph_dir):
    with open(graph_dir + '{}nodes.txt'.format(D), 'r') as f:
        list1 = f.readlines()
    f.close()
    num_edges = int(list1[0].split()[1])
    edges = np.zeros([len(list1)-1, 2], dtype=int)
    for i in range(len(list1)-1):
        edges[i] = list1[i+1].split()
    neighbors = {}.fromkeys(np.arange(D))
    for key in neighbors.keys():
        neighbors[key] = []
    for edge in edges:
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])
    '''
    for key in neighbors.keys():
        neighbors[key] = np.array(neighbors[key])
    '''
    J = np.loadtxt(graph_dir + 'Jij{}nodes.txt'.format(D), dtype=np.float64)

    return num_edges, edges, neighbors, J


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    #    torch.set_num_threads(8)

    device = torch.device("cpu" if args.cuda < 0 else "cuda:" + str(args.cuda))
    if args.graph == 'rrg' or args.graph == 'rer':
        graph = nx.random_regular_graph(args.k, args.n, seed=args.seed)
        edges = graph.edges
        print("regular random graph, n=", args.n, "k=", args.k, "seed=", args.seed, "maxdim=", args.maxdim)
    elif args.graph == 'gnp' or args.graph == 'ran' or args.graph == 'er':
        graph = nx.gnp_random_graph(args.n, 1.0 * args.c / args.n, seed=args.seed)
        edges = list(graph.edges)
        print("ER random graph, n=", args.n, "c=", args.c, "seed=", args.seed, "maxdim=", args.maxdim)
    elif args.graph == 'line':
        edges = [(i, i + 1) for i in range(args.n - 1)]
        print("line graph, n=", args.n, "seed=", args.seed, "maxdim=", args.maxdim)
    elif args.graph == '2dsquare':
        graph = nx.grid_2d_graph(args.n, args.n)
        graph = nx.Graph(graph)
        edges_2d = list(graph.edges)
        edges = [(i[0] * args.n + i[1], j[0] * args.n + j[1]) for i, j in edges_2d]
        args.L = args.n
        args.n = args.n ** 2
        print("2d lattice, L=", args.L, "seed=", args.seed, "maxdim=", args.maxdim)
    elif args.graph == 'c60':
        A = np.loadtxt('c60.E', dtype=np.int32)
        edges = []
        for i in range(60):
            edges.append([i, A[i, 0] - 1])
            edges.append([i, A[i, 1] - 1])
            edges.append([i, A[i, 2] - 1])
    elif args.graph == 'tree':
        graph = nx.random_tree(args.n, seed=args.seed)
        edges = list(graph.edges)
        print(edges)
    elif args.graph == 'complete':
        graph = nx.complete_graph(args.n)
        edges = list(graph.edges)
    elif args.graph == 'scale_free':
        graph=nx.barabasi_albert_graph(args.n,args.m,seed=args.seed)
        edges=list(graph.edges)
    elif args.graph == 'sw':
        graph=nx.watts_strogatz_graph(args.n, args.k, args.p, seed=args.seed)
        edges=list(graph.edges)
    elif args.graph == 'rrgn300k4':
        args.beta = 0.8
        args.n = 300
        _, edges, _, Jraw = readgraph(args.n, '../graph/')
        Jraw = torch.from_numpy(Jraw).to(torch.float64).to(device)
        weights = Jraw[edges.transpose()]
        weights.requires_grad = True
        args.Jij = None
        args.field = 'zero'
    elif args.graph == 'read_from_file':
        valtype2,w_file,h_file,n,edges=read_graph_from_file(args.file)
        args.n=n
    
    np.random.seed(args.seed)
    edges = np.unique(np.array([sorted(a) for a in edges]), axis=0)
    idx_i,idx_j=edges[1]
    spin=np.ones(args.n)
    spin[idx_i]=-1
    spin[idx_j]=-1
    spin1=np.ones(args.n)
    spin1[idx_i]=1
    spin1[idx_j]=-1
    valtype=2*np.ones([args.n],int)
    valtype1=2*np.ones([args.n],int)
    valtype[idx_i]=1
    valtype[idx_j]=1
    if args.Jij == 'ferro':
        weights = np.ones(len(edges))
    elif args.Jij == 'rand':
        weights = np.random.rand(len(edges))
    elif args.Jij == 'randn':
        weights = np.random.randn(len(edges))
    elif args.Jij == 'sk':
        weights = np.random.randn(len(edges)) / np.sqrt(args.n)
    elif args.Jij == 'binary':
        weights = np.random.randint(0, 2, len(edges)) * 2 - 1
    elif args.Jij == 'normal':
        weights = np.random.normal(0,1/args.L,len(edges))
    #print(weights)
    if args.field == 'zero':
        fields = np.zeros(args.n)
    elif args.field == 'one':
        fields = np.ones(args.n)
    elif args.field == 'rand':
        fields = np.random.rand(args.n)
    elif args.field == 'randn':
        fields = np.random.randn(args.n)
    elif args.field=='normal':
        fields=np.random.normal(0,1/args.L,args.n)
    

    fields = fields * args.gamma

    G = nx.Graph()
    G.add_nodes_from(np.arange(args.n))
    G.add_edges_from(edges)
    G_backup = G.copy()
    t0 = time.time()
    if args.seed2 < 0:
        args.seed2 = args.seed

    beta = torch.tensor([args.beta], dtype=torch.float64, device=device)
    J = torch.zeros(args.n, args.n, dtype=torch.float64, device=device)
    idx = np.array(edges)
    W = torch.tensor(weights, dtype=torch.float64, device=device)
    J[idx[:, 0], idx[:, 1]] = W
    J[idx[:, 1], idx[:, 0]] = W
    #print(J)
    H = torch.tensor(fields, dtype=torch.float64, device=device)
    h=[]
    h1=[]
    for i in range (args.n):
        if i!=idx_i and i!=idx_j:
           h.append(torch.exp(H[i] *beta* torch.tensor([1, -1], dtype=torch.float64, device=device)))
           h1.append(torch.exp(H[i] * beta*torch.tensor([1, -1], dtype=torch.float64, device=device)))
        else:
           h.append(torch.exp(H[i] * beta*torch.tensor([spin[i]], dtype=torch.float64, device=device)))
           h1.append(torch.exp(H[i] * beta*torch.tensor([spin1[i]], dtype=torch.float64, device=device)))
		   
    w=[]
    w1=[]
    for edge in range (len(edges)):
        m,n=edges[edge]
        if m==idx_i and n==idx_j:
           w.append(torch.exp(W[edge] * beta * torch.tensor([spin[m]*spin[n]], dtype=torch.float64, device=device)).reshape(1,1))
           w1.append(torch.exp(W[edge] * beta * torch.tensor([spin1[m]*spin1[n]], dtype=torch.float64, device=device)).reshape(1,1))
        elif m==idx_i and n!=idx_j:
           w.append(torch.exp(W[edge] * beta * torch.tensor([spin[m],-spin[m]], dtype=torch.float64, device=device)).reshape(1,2))
           w1.append(torch.exp(W[edge] * beta * torch.tensor([spin1[m],-spin1[m]], dtype=torch.float64, device=device)).reshape(1,2))
        elif m==idx_j:
           w.append(torch.exp(W[edge] * beta * torch.tensor([spin[m],-spin[m]], dtype=torch.float64, device=device)).reshape(1,2))
           w1.append(torch.exp(W[edge] * beta * torch.tensor([spin1[m],-spin1[m]], dtype=torch.float64, device=device)).reshape(1,2))
        elif n==idx_i:
           w.append(torch.exp(W[edge] * beta * torch.tensor([spin[n],-spin[n]], dtype=torch.float64, device=device)).reshape(2,1))
           w1.append(torch.exp(W[edge] * beta * torch.tensor([spin1[n],-spin1[n]], dtype=torch.float64, device=device)).reshape(2,1))		   
        elif m!=idx_i and n==idx_j:
           w.append(torch.exp(W[edge] * beta * torch.tensor([spin[n],-spin[n]], dtype=torch.float64, device=device)).reshape(2,1))
           w1.append(torch.exp(W[edge] * beta * torch.tensor([spin1[n],-spin1[n]], dtype=torch.float64, device=device)).reshape(2,1))
        else:
           w.append(torch.exp(W[edge] * beta * torch.tensor([[1, -1], [-1, 1]],
                                                                  dtype=torch.float64, device=device)))
           w1.append(torch.exp(W[edge] * beta * torch.tensor([[1, -1], [-1, 1]],
                                                                  dtype=torch.float64, device=device)))																  
    if (args.graph=='read_from_file'):
      
      w=w_file
      h=h_file
      valtype=vartype2																  
    '''
    if args.raw:
        args.node = "raw"
    '''
    
    tn = Tensor_Network(args.n, valtype1,edges, W, H, beta, seed=args.seed2, maxdim=args.maxdim,
                        verbose=args.verbose, Dmax=args.Dmax, chi=args.chi, node_type=args.node)
    #tn_ij_1=Tensor_Network(args.n, valtype,edges, w, h, beta, seed=args.seed2, maxdim=args.maxdim,
                        #verbose=args.verbose, Dmax=args.Dmax, chi=args.chi, node_type=args.node)
    #tn_ij_2=Tensor_Network(args.n, valtype,edges, w1, h1, beta, seed=args.seed2, maxdim=args.maxdim,
                        #verbose=args.verbose, Dmax=args.Dmax, chi=args.chi, node_type=args.node)
    t0 = time.time()
    lnZ_tn = tn.contraction()
    time_tn=time.time()-t0
    if args.backward: 
       (lnZ_tn / beta).backward()
    lnZ_tn = lnZ_tn / tn.n
    print("lnZ_tn = {:.15g}, time: {:.2g} Sec. maxdim_inter={:d}".format(lnZ_tn.item(), time.time() - t0,
                                                                         int(tn.maxdim_intermediate)))
    print("free energy ={:.15g}".format(-lnZ_tn.item()/args.beta))
    if args.graph == 'rrgn300k4':
        print("F = {:.15g}".format(-lnZ_tn.item() / args.beta))

    if args.graph == '2dsquare':
        from exact import kacward

        t0 = time.time()
        exact_solution = kacward(args.L, J, args.beta)
        lnZ_exact = exact_solution.lnZ / args.L ** 2
        print("lnZ_Exact_kacward = {:.15g}, time: {:.2g} Sec.".format(lnZ_exact, time.time() - t0))
        print("Error of lnZ: %.3g" % (lnZ_tn - lnZ_exact))
    if args.fvsenum:
        from exact import exact

        t0 = time.time()
        exact1 = exact(G_backup, J, H,args.beta, device, args.seed)
        lnZ_exact = exact1.lnZ_fvs() / len(tn.tensors)
        print("lnZ_Exact = {:.15g}, Free energy_Exact={:.15g}, time: {:.2g} Sec.".format(lnZ_exact,-lnZ_exact/args.beta, time.time() - t0))
        print("Error of lnZ: %.3g" % (lnZ_tn - lnZ_exact))
        print("Error of free energy: %.3g" % -(lnZ_tn - lnZ_exact)/args.beta)
        

    if args.mf:
       mf=MeanField(G_backup,J,H,args.beta,device)
       t0=time.time()
       fe_BP, energy_BP, entropy_BP, mag_BP, correlation_BP, step=mf.BP()
       time_bp=time.time()-t0
       t0=time.time()
       F_tap, E_tap, S_tap,iter_count_tap=mf.F_tap(0.3)
       time_tap=time.time()-t0
       t0=time.time()
       F_nmf, E_nmf, S_nmf,iter_count_nmf=mf.F_nmf(0.3)
       time_nmf=time.time()-t0
      
    if args.backward:
      correlation_tn = W.grad
#      print('entropy: ', (-beta ** 2 * beta.grad).item())
#      print('energy: ', (-(lnZ_tn * tn.n) / beta - beta * beta.grad).item())
#      print(correlation_tn)
#      print(H.grad)
#      print(edges)
    if args.fvsenum: 
       F_exact=-lnZ_exact/args.beta
    F_tn=-lnZ_tn/args.beta
    print(F_tn)
    with open('{}_{}_Dmax={}_chi={}_Jij={}.txt'.format(args.graph,args.n,args.Dmax,args.chi,args.Jij), 'a') as fp:
            #f.write('{} {}\n'.format(args.n, len(edges)))
            #fp.write('{}  {:.15g} {:.15g} {:.3g}\n'.format(args.n ,lnZ_exact, lnZ_tn - lnZ_exact,time_tn))
            if args.fvsenum:
              fp.write('{} {:.15g} {:.15g} {:.3g}  '.format(args.beta,F_exact, (F_tn-F_exact).item(),time_tn))
            else:
              fp.write('{} {:.15g} {:.15g} {:.3g}\n  '.format(args.beta,args.beta, (F_tn).item(),time_tn))
            if args.mf:
              fp.write('{:.15g} {:.3g} {:.15g} {:.3g} {:.15g} {:.3g} '.format(F_nmf-F_exact,time_nmf,F_tap-F_exact,time_tap,fe_BP-F_exact,time_bp))
              fp.write('{} {} {}\n'.format(iter_count_nmf,iter_count_tap,step))
    fp.close()



