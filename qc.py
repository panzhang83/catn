"""
dsfad
"""
#import torch
import numpy as np
import math
import networkx as nx
import string
import time
import sys
from tensor_network_np import Tensor_Network_np
from peps import PEPS
import convert


if __name__ == '__main__':
    import argparse
#    torch.set_num_threads(8)
    print("analysing args")
    t_begin = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-n", type=int, default=30, help="number of nodes")
    parser.add_argument("-lx", type=int, default=-1, help="Lx")
    parser.add_argument("-ly", type=int, default=-1, help="Ly")
    parser.add_argument("-k", type=int, default=3, help="degree, integer")
    parser.add_argument("-c", type=float, default=3.0, help="average degree in gnp")
    parser.add_argument("-beta", type=float, default=0.8, help="beta")
    parser.add_argument("-gamma", type=float, default=1.0, help="external fields strength")
    parser.add_argument("-cutoff", type=float, default=1.0e-14, help="Cut off")
    parser.add_argument("-maxdim", type=int, default=30, help="maximum dimension of intermediate tensor")

    parser.add_argument("-seed", type=int, default=1, help="seed")
    parser.add_argument("-seed2", type=int, default=-1, help="seed2")
    parser.add_argument("-bins", type=int, default=20, help="number of output")
    parser.add_argument("-graph", type=str, default='qft20', help="graph")
    parser.add_argument("-Jij", type=str, default='ferro', choices=['ferro','rand', 'randn','sk'])
    parser.add_argument("-field", type=str, default='zero', choices=['one','rand', 'randn'])
    parser.add_argument("-node", type=str, default='np',choices=['mps','peps','np'], help="node representation, raw or mps")
    parser.add_argument("-cuda", type=int, default=-1, help="GPU #")
    parser.add_argument("-verbose", type=int, default=-1, help="verbose")
    parser.add_argument("-raw", action='store_true', help="Node type set to 'raw'")
    parser.add_argument("-Dmax", type=int, default=12, help="Maximum physical bond dimension of the tensors. With Dmax<0, contraction will be exact")
    parser.add_argument("-chi", type=int, default=32, help="Maximum virtual bond dimension of the mps.")
    parser.add_argument("-norm", type=int, default=1, choices=[0,1,2],help="normalization methods")
    parser.add_argument("-svdopt", type=int,default=1,choices=[0,1], help="optimize svd of two 3-way tensors")
    parser.add_argument("-select", type=int,default=1,choices=[0,1,2], help="Heuristic for selecting edges in contractions")
    parser.add_argument("-reverse", type=int,default=1,choices=[0,1], help="whether reverse the mps?")
    parser.add_argument("-swapopt", type=int,default=1,choices=[0,1], help="optimize swap() operations")
    parser.add_argument("-fvsenum", action='store_true', help="compute exact solution by enumerating configurations of feedback set")
    args = parser.parse_args()

#    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))
    if(args.seed2<0):
        args.seed2 = args.seed
    dirname = './'
#    dirname = '/home/pan/work/gitlab/qc/tensorqc/'
#    if(args.node !="peps"):
#        name= dirname + 'graphs/'+args.graph
#    else:
#    name= '/home/pan/work/gitlab/qc/tensorqc/graphs/'+args.graph+'-rank4'
    name= dirname + 'graphs/'+args.graph
    print("graph",name)

    tensors, labels_einsum = convert.convert_to_incidence_list(name)
    labels = convert.incidence_list_to_adjacency_list(labels_einsum)

#    fname = name+".labels.dat"
#    lines = open(fname,"r").readlines()
#    lines = [i.split() for i in lines]
#    print(lines)
#    dic = ['0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#    print(labels_einsum)
#    seq = ''
#    for i in labels_einsum:
#        for j in i:
#            seq = seq + dic[j]
#        seq = seq+','
#    seq = seq[:-1]
#    print([[list(tensors[i].shape),seq.split(",")[i]] for i in range(len(tensors))])
#    print(seq)
#    psi = np.einsum(seq,*tensors)
#
#    print("psi=",psi,"prob=",np.abs(psi)**2)
#    input("***")


#    torch.manual_seed(args.seed)
    svdopt = True if args.svdopt==1 else False
    reverse = True if args.reverse==1 else False
    swapopt = True if args.swapopt==1 else False
#    print(labels)
    print("init peps")
    if(args.lx>0 and args.ly<0):
        args.ly=args.lx
    if args.node == "peps" or args.node == "exact":
        tn = PEPS(args.lx,args.ly,tensors,labels,seed=args.seed2,mydevice='cpu',maxdim=args.maxdim,verbose=args.verbose,Dmax=args.Dmax,chi=args.chi, node_type = args.node,cutoff=args.cutoff)
    elif args.node == "np" or args.node=="mps":
        tn = Tensor_Network_np(tensors,labels,seed=args.seed2,mydevice='cpu',maxdim=args.maxdim,verbose=args.verbose,Dmax=args.Dmax,chi=args.chi, node_type = args.node,norm_method = args.norm,svdopt = svdopt, swapopt = swapopt,reverse=reverse,bins = args.bins,cutoff=args.cutoff,select=args.select)

    t0 = time.time()
    lnZ_tn,error,psi = tn.contraction()
    print("lnZ_tn = {:.15g}, prob={:.8g}, svd_error = {:.15g}, time: {:.2g} Sec. maxdim_inter={:d}".format(lnZ_tn.item(),(np.exp(lnZ_tn)**2).item(),np.linalg.norm(error), time.time() - t0,int(tn.maxdim_intermediate)))
    psi=np.exp(lnZ_tn)*psi
    print("psi=",psi,"prob=",np.abs(psi)**2)
    print("time",time.time()-t_begin,"Seconds")
    print("CMD: "," ".join(["python"]+sys.argv))

