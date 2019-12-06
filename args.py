import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument("-n", type=int, default=30, help="number of nodes")
parser.add_argument("-k", type=int, default=3, help="degree, integer")
parser.add_argument("-c", type=float, default=3.0, help="average degree in gnp")
parser.add_argument("-beta", type=float, default=0.8, help="beta")
parser.add_argument("-gamma", type=float, default=1.0, help="external fields strength")
parser.add_argument("-maxdim", type=int, default=30, help="maximum dimension of intermediate tensor")

parser.add_argument("-seed", type=int, default=1, help="seed")
parser.add_argument("-seed2", type=int, default=-1, help="seed2")
parser.add_argument("-graph", type=str, default='rrg', help="graph")
parser.add_argument("-Jij", type=str, default='ferro',
                    choices=['ferro', 'rand', 'randn', 'sk', 'binary','normal'])
parser.add_argument("-field", type=str, default='zero', choices=['one', 'rand', 'randn','normal'])
parser.add_argument("-node", type=str, default='mps', choices=['raw', 'mps'],
                    help="node representation, raw or mps")
parser.add_argument("-bins", type=int, default=20, help="number of output")
parser.add_argument("-cutoff", type=float, default=1.0e-15, help="Cut off")
parser.add_argument("-m",type=int,default=3,help="Number of edges to attach for scale free graph")
parser.add_argument("-p",type=float,default=0.3,help="probability of an edge tobe rewire")
parser.add_argument("-cuda", type=int, default=-1, help="GPU #")
parser.add_argument("-verbose", type=int, default=-1, help="verbose")
parser.add_argument("-corder", action='store_true', help="Node type set to 'raw'")
parser.add_argument('-compress',action='store_true',help="compress all mps")
parser.add_argument('-cut_bond',action='store_true',help="cut_bond_option")
parser.add_argument("-norm", type=int, default=1, choices=[0,1,2],help="normalization methods")
parser.add_argument("-svdopt", type=int,default=1,choices=[0,1], help="optimize svd of two 3-way tensors")
parser.add_argument("-select", type=int,default=1,choices=[0,1,2], help="Heuristic for selecting edges in contractions")
parser.add_argument("-reverse", type=int,default=1,choices=[0,1], help="whether reverse the mps?")
parser.add_argument("-swapopt", type=int,default=1,choices=[0,1], help="optimize swap() operations")
parser.add_argument("-mf", action='store_true', help="use mean fields to calculate freeenergy")
parser.add_argument("-backward",action='store_true',help="use backward to calculate correlation")
parser.add_argument("-Dmax", type=int, default=32,
                    help="Maximum physical bond dimension of the tensors. With Dmax<0, contraction will be exact")
parser.add_argument("-chi", type=int, default=32, help="Maximum virtual bond dimension of the mps.")
parser.add_argument("-fvsenum", action='store_true',
                    help="compute exact solution by enumerating configurations of feedback set")
parser.add_argument("-calc_mag", action='store_true',
                    help="calculate magnetism by pin nodes")
parser.add_argument("-calc_cor", action='store_true',
                    help="calculate correlation by pin nodes")
args = parser.parse_args()
