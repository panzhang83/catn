"""
Computing LnZ of a graphical model using tensor networks
## Command line examples: 
* regular random graph with degree 3:
    python lnz.py -n50 -k 3 -fvsenum
    python lnz.py -n160 -k 3 -beta 0.9 -seed 3 -maxdim 34

* football c60 graph
    python lnz.py -beta 0.45 -seed 3 -maxdim 30 -verbose 1 -fvsenum -graph c60 # football graph
* 2d ferromagnetic Ising model
    python lnz.py -n20 -beta 0.45 -seed 3 -maxdim 30 -graph 2dsquare -verbose 1

python lnz.py -n10 -beta 0.45 -seed 3 -maxdim 32 -graph 2dsquare -verbose 1  -Dmax -1 -chi 100
python lnz.py -n10 -beta 0.45 -seed 3 -maxdim 32 -graph 2dsquare -verbose 1  -Dmax -1 -chi 100 -Jij randn -field randn

TODO:
    1. Optimize select_edge_total_dimension(). Storing total dimension as a variable, manitainig a sorted list for select edges.
"""

import torch
import numpy as np
import math
import networkx as nx
import string
import time
import sys
from tn_np import Tensor_Network_np
from args import args
from bp_mf import MeanField
import random


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
        '''
        G = nx.Graph()
        G.add_nodes_from(np.arange(args.n ** 2))
        for i in range(8, args.n - 8):
            for j in range(8, args.n - 8):
                if j < args.n - 1 - 8:
                    G.add_edge(i * args.n + j, i * args.n + j + 1)
                if i >= 1 + 8:
                    G.add_edge(i * args.n + j, (i - 1) * args.n + j)
        edges = list(G.edges)
        '''
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
    elif args.graph == 'rrgn300k4':
        args.beta = 0.8
        args.n = 300
        _, edges, _, Jraw = readgraph(args.n, '../graph/')
        weights = Jraw[edges.transpose()]
        weights.requires_grad = True
        args.Jij = None
        args.field = 'zero'
    elif args.graph == 'from_file':
        _, edges, _, Jraw = readgraph(args.n, '../graph/')
        weights = Jraw[edges.transpose()]
        args.Jij = None
        args.field = 'zero'
    elif args.graph == 'scale_free':
        graph=nx.barabasi_albert_graph(args.n,args.m,seed=args.seed)
        edges=list(graph.edges)
    elif args.graph == 'sw':
        graph=nx.watts_strogatz_graph(args.n, args.k, args.p, seed=args.seed)
        edges=list(graph.edges)
    random.seed(args.seed)
    np.random.seed(args.seed)
    new_order=np.arange(len(edges))
    random.shuffle(new_order)
    edges = np.unique(np.array([sorted(a) for a in edges]), axis=0)
    #edges=edges[new_order]
    print(edges)
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

    if args.field == 'zero':
        fields = np.zeros(args.n)
    elif args.field == 'one':
        fields = np.ones(args.n)
    elif args.field == 'rand':
        fields = np.random.rand(args.n)
    elif args.field == 'randn':
        fields = np.random.randn(args.n)

    fields = fields * args.gamma

    G = nx.Graph()
    G.add_nodes_from(np.arange(args.n))
    G.add_edges_from(edges)
    G_backup = G.copy()
    # t0 = time.time()
    if args.seed2 < 0:
        args.seed2 = args.seed

    J = np.zeros([args.n, args.n], dtype=np.float64)
    H = torch.tensor(fields, dtype=torch.float64)
    idx = np.array(edges)
    J[idx[:, 0], idx[:, 1]] = weights
    J[idx[:, 1], idx[:, 0]] = weights
    svdopt = True if args.svdopt==1 else False
    reverse = True if args.reverse==1 else False
    swapopt = True if args.swapopt==1 else False
    tn = Tensor_Network_np(args.n, edges, weights, args.beta * fields, args.beta, seed=args.seed2, maxdim=args.maxdim,
                        verbose=args.verbose, Dmax=args.Dmax, chi=args.chi, node_type=args.node,norm_method = args.norm,svdopt = svdopt, swapopt = swapopt,reverse=reverse,bins = args.bins,select=args.select)
    # tn.tensors[0].tensor.norm().backward()
    t0 = time.time()
    lnZ_tn, error, psi = tn.contraction()
    lnZ_tn = lnZ_tn / tn.n
    time_tn=time.time()-t0
    print("lnZ_tn = {:.15g}, time: {:.2g} Sec. maxdim_inter={:d}".format(lnZ_tn.item(), time.time() - t0,
                                                                         int(tn.maxdim_intermediate)))
    print("free energy ={:.15g}".format(-lnZ_tn.item()/args.beta))
    if args.graph == 'rrgn300k4':
        print("F = {:.15g}".format(-lnZ_tn.item() / args.beta))

    if args.graph == '2dsquare' and args.field == 'zero':
        from exact import kacward

        t0 = time.time()
        exact_solution = kacward(args.L, J, args.beta)
        lnZ_exact = exact_solution.lnZ / args.L ** 2
        print("lnZ_Exact_kacward = {:.15g}, time: {:.2g} Sec.".format(lnZ_exact, time.time() - t0))
        print("Error of lnZ: %.3g" % (lnZ_tn - lnZ_exact))
    if args.fvsenum:
        from exact import exact

        t0 = time.time()
        exact1 = exact(G_backup, torch.from_numpy(J), torch.from_numpy(fields), args.beta, 'cpu', args.seed)
        lnZ_exact = exact1.lnZ_fvs() / len(tn.tensors)
        print("lnZ_Exact = {:.15g}, time: {:.2g} Sec.".format(lnZ_exact, time.time() - t0))
        print("Error of lnZ: %.3g" % (lnZ_tn - lnZ_exact))
        #correlation, edges = exact1.correlation()
        #mag = exact1.magnetization()
        #print(correlation, edges)
        #print(mag)

    if args.calc_mag:
        lnZ_mag = np.empty([args.n, 2], dtype=np.float64)
        for key in range(args.n):
            for value in [0, 1]:
                tn.G.add_nodes_from(np.arange(args.n))
                tn.G.add_edges_from(edges)
                tn.construct_tensor(key, value)
                tn.select_edge_init()
                lnZ_mag[key, value], _, _ = tn.contraction()

        lnZ_mag -= (lnZ_tn * args.n)
        p_mag = np.exp(lnZ_mag)
        p_mag[:, 0] *= -1
        mag_forward = np.sum(p_mag, axis=1)
        print(mag_forward)
    if args.calc_cor:
        lnZ_cor = np.empty([len(edges), 4], dtype=np.float64)
        for key in range(len(edges)):
            m, n = edges[key]
            for value1 in [0, 1]:
                for value2 in [0, 1]:
                    tn.G.add_nodes_from(np.arange(args.n))
                    tn.G.add_edges_from(edges)
                    tn.construct_tensor(m, value1, n, value2)
                    tn.select_edge_init()
                    lnZ_cor[key, value1 * 2 + value2], _, _ = tn.contraction()
        lnZ_cor -= (lnZ_tn * args.n)
        p_cor = np.exp(lnZ_cor)
        p_cor[:, 1] *= -1
        p_cor[:, 2] *= -1
        cor_forward = np.sum(p_cor, axis=1)
        print(cor_forward)
    if args.mf:
       mf=MeanField(G_backup,torch.from_numpy(J),H,args.beta,device)
       t0=time.time()
       fe_BP, energy_BP, entropy_BP, mag_BP, correlation_BP, step=mf.BP()
       time_bp=time.time()-t0
       t0=time.time()
       F_tap, E_tap, S_tap,iter_count_tap=mf.F_tap(0.3)
       time_tap=time.time()-t0
       t0=time.time()
       F_nmf, E_nmf, S_nmf,iter_count_nmf=mf.F_nmf(0.3)
       time_nmf=time.time()-t0
    if args.fvsenum: 
       F_exact=-lnZ_exact/args.beta
    F_tn=-lnZ_tn/args.beta
    print(F_tn)
    with open('np_{}_{}_Dmax={}_chi={}_Jij={}.txt'.format(args.graph,args.n,args.Dmax,args.chi,args.Jij), 'a') as fp:
            #f.write('{} {}\n'.format(args.n, len(edges)))
            #fp.write('{}  {:.15g} {:.15g} {:.3g}\n'.format(args.n ,lnZ_exact, lnZ_tn - lnZ_exact,time_tn))
            if args.fvsenum:
              fp.write('{} {:.16g} {:.17g} {:.3g}  '.format(args.beta,F_exact, (F_tn-F_exact).item(),time_tn))
            else:
              fp.write('{} {:.16g} {:.17g} {:.3g}\n  '.format(args.n,args.beta, (F_tn).item(),time_tn))
            if args.mf:
              fp.write('{:.15g} {:.3g} {:.15g} {:.3g} {:.15g} {:.3g} '.format(F_nmf-F_exact,time_nmf,F_tap-F_exact,time_tap,fe_BP-F_exact,time_bp))
              fp.write('{} {} {}\n'.format(iter_count_nmf,iter_count_tap,step))
    fp.close()