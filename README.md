# catn
Contracting Arbitrary Tensor Networks

This repo contains a numpy and pytorch implementation of the algorithm proposed in arXiv:~~~
for approximately contracting arbitrary tensor networks. Two examples are included for demonstration the performance of our algorithm.

# Examples for computing free energy of graphical models:
* Spin glasses on a regular random graph with degree 3:
python lnz_np.py -n 50 -beta 0.1 -k 3 -graph rrg  -seed 1 -Dmax 20 -chi 500  -Jij randn  -fvs -mf

* 2d ferromagnetic Ising model
python lnz_np.py -n 10  -beta 0.2 -graph 2dsquare -select 1 -seed 1 -Dmax 50 -chi 500  -node mps -Jij ferro -reverse 1 -compress

* Sherrington-Kirkpatrick model with 20 spins
python lnz_np.py -n 20  -beta 0.2 -graph complete -seed 1 -Dmax 50 -chi 500 -Jij sk -reverse 0 -compress -fvs -mf

* Spin glasses on a small-world network
python lnz_np.py -n 50  -beta 0.2  -k 4 -graph sw -select 1 -seed 1 -Dmax 50 -chi 500  -node mps  -Jij randn -reverse 1 -compress -fvs -mf

* Contract a circuit with $10\times 10=100$ qubits, with depth (1+8+1), as stored in graphs/supremacy2d100-d10-seed1-rank3-new...., with physical bond dimension $D=600$ and inner bond dimension $\chi=1500$
First run `pip install fire`, then
python ./qc.py -graph supremacy2d100-d10-seed1-rank3-new -Dmax 600 -chi 1500
