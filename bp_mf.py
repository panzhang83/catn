import numpy as np
import networkx as nx
import torch
def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))
class MeanField():
    def __init__(self, graph,J,H,beta,mydevice):
        self.J=J
        self.H=H
        self.conv_crit = 1e-6
        self.max_iter = 2*10**3
        self.beta=beta
        self.C_model=[]
        self.n=J.shape[0]
        self.graph=graph
        self.device = mydevice
    def get_entropy_fact(self, m):
        return -torch.sum((1 + m) / 2 * torch.log((1 + m) / 2) +
                          (1 - m) / 2 * torch.log((1 - m) / 2))

    def get_free_energy_nmf(self, m):
        S = self.get_entropy_fact(m) / self.n
        E = (-1 / 2 * m @ self.J @ m +self.H@ m)/ self.n
        F = E - S / self.beta
        return F, E, S

    def F_nmf(self, damping=0.3):
        m = torch.tanh(torch.randn([self.n], dtype=torch.float64,device=self.device))
        for iter_count in range(self.max_iter):
            m_new = (damping * m +
                     (1 - damping) * torch.tanh(self.beta * self.J @ m+self.beta * self.H))
            diff = (m_new - m).norm()
            if diff < self.conv_crit:
                break
            m = m_new
        else:
            print('conv_crit not meet, diff = {}'.format(diff))

        F, E, S = self.get_free_energy_nmf(m)
        print('NMF:\tF = {:.15g}\tE = {:.15g}\tS = {:.15g}\titer = {}'.format(
            F, E, S, iter_count))

        return F, E, S,iter_count

    def get_free_energy_tap(self, m):
        S = self.get_entropy_fact(m) / self.n
        E = (-1 / 2 * m @ self.J @ m + self.H @ m )/ self.n
        G2 = -self.beta / 4 * (1 - m**2) @ (self.J**2) @ (1 - m**2) / self.n
        E += G2 
        F = E - S / self.beta
        return F, E, S

    def F_tap(self, damping=0.3):
        m = torch.tanh(torch.randn([self.n],dtype=torch.float64, device=self.device))
        for iter_count in range(self.max_iter):
            m_new = (damping * m + (1 - damping) *
                     torch.tanh(self.beta * self.J @ m + self.beta * self.H -m *
                                (self.beta * self.J)**2 @ (1 - m**2)))
            diff = (m_new - m).norm()
            if diff < self.conv_crit:
                break
            m = m_new
        else:
            print('conv_crit not meet, diff = {}'.format(diff))

        F, E, S = self.get_free_energy_tap(m)
        print('TAP:\tF = {:.15g}\tE = {:.15g}\tS = {:.15g}\titer = {}'.format(
            F, E, S, iter_count))

        return F, E, S,iter_count
    
    
    def BP(self):
        stepmax = 1000
        epsilon = 1e-6
        difference_max = 10
        damping_factor = 0
        beta=self.beta
        num_edges=len(list(self.graph.edges()))
        neighbors=[]
        for i in range(self.n):
             neighbors.append(list(self.graph.adj[i]))
        edges=list(self.graph.edges())
      
        J=self.J.detach().numpy()
      
        D=self.n
       
       
        h = np.random.randn(D, D)
        # belief propagation
        for step in range(stepmax):
          for i in range(D):
            for j in range(len(neighbors[i])):
                a = neighbors[i][j]
                B = list(neighbors[i])
                B.remove(a)
                temp = (np.arctanh(
                        np.tanh(beta * J[i, B]) * np.tanh(beta * h[B, i])
                        ) / (beta)).sum()
                temp = damping_factor*h[i][a] + (1-damping_factor)*temp
                difference = abs(temp - h[i][a])
                h[i][a] = temp
                if i == 0 and j == 0:
                    difference_max = difference
                elif difference > difference_max:
                    difference_max = difference
          if difference_max <= epsilon:
             break
   
        # calculate free energy
        fe_node = np.zeros(D)
        for i in range(D):
           B = list(neighbors[i])
           temp1 = (np.cosh(beta * (J[i, B] + h[B, i])) /
                np.cosh(beta * h[B, i])).prod()
           temp2 = (np.cosh(beta * (-J[i, B] + h[B, i])) /
                np.cosh(beta * h[B, i])).prod()
           fe_node[i] = - np.log(temp1 + temp2) / beta
        fe_node_sum = np.sum(fe_node)

        fe_edge = np.zeros(num_edges)
        edge_count = 0
        for edge in edges:
           i, j = edge
           temp1 = np.exp(beta*J[i,j]) * np.cosh(beta*(h[i,j]+h[j,i])) + \
                np.exp(-beta*J[i,j]) * np.cosh(beta*(h[i,j]-h[j,i]))
           temp2 = 2*np.cosh(beta*h[i,j])*np.cosh(beta*h[j,i])
           fe_edge[edge_count] = - np.log(temp1/temp2) / beta
           edge_count += 1
        fe_edge_sum = np.sum(fe_edge)

        fe_sum = fe_node_sum - fe_edge_sum

        # calculate energy
        energy_BP = np.zeros(num_edges)
        edge_count = 0
        for edge in edges:
           i, j = edge
           temp1 = -J[i,j]*np.exp(beta*J[i,j])*np.cosh(beta*(h[i,j]+h[j,i])) + \
                J[i,j]*np.exp(-beta*J[i,j])*np.cosh(beta*(h[i,j]-h[j,i]))
           temp2 = np.exp(beta*J[i,j])*np.cosh(beta*(h[i,j]+h[j,i])) + \
                np.exp(-beta*J[i,j])*np.cosh(beta*(h[i,j]-h[j,i]))
           energy_BP[edge_count] = temp1 / temp2
           edge_count += 1
        energy_BP = np.sum(energy_BP)

        # calculate entropy
        entropy_BP = beta*(energy_BP - fe_sum)

        # calcualte magnetzation
        mag_BP = np.zeros(D)
        for i in range(D):
           B = list(neighbors[i])
           temp = np.arctanh(
                np.tanh(beta*J[i, B]) * np.tanh(beta*h[B,i])
                ).sum()
           mag_BP[i] = np.tanh(temp)

        # calculate connected correlation
        correlation_BP = np.empty(num_edges)
        edge_count = 0
        for edge in edges:
           i, j = edge
           temp1 = np.exp(beta*J[i,j])*np.cosh(beta*(h[i,j]+h[j,i]))
           temp2 = np.exp(-beta*J[i,j])*np.cosh(beta*(h[i,j]-h[j,i]))
           correlation_BP[edge_count] = (temp1 - temp2) / (temp1 + temp2) - \
           mag_BP[i] * mag_BP[j]
           edge_count += 1
        print('BP:\tF = {:.15g}\tE = {:.15g}\tS = {:.15g}\titer = {}'.format(
            fe_sum/D, energy_BP/D, entropy_BP/D,step))
        return fe_sum/D, energy_BP/D, entropy_BP/D, mag_BP, correlation_BP, step
if __name__ =='__main__'  :
    n=60
    graph = nx.random_regular_graph(3, n, seed=1)
    edges = graph.edges
    edges = np.unique(np.array([a for a in edges]), axis=0)
    np.random.seed(1)
    device = torch.device("cpu" )
    weights = np.random.randn(len(edges))
    print(weights)
    fields = np.zeros(n)
	
    J = torch.zeros(n, n, dtype=torch.float64)
    idx = np.array(edges)
   
    W = torch.tensor(weights, dtype=torch.float64)
    J[idx[:, 0], idx[:, 1]] = W
    J[idx[:, 1], idx[:, 0]] = W
    H = torch.tensor(fields, dtype=torch.float64,  requires_grad=True)
    beta=1
     
    mf=MeanField(graph,J,H,beta,device)
    fe_sum, energy_BP, entropy_BP, mag_BP, correlation_BP, step=mf.BP()
	
    F, E, S,iter=mf.F_tap(0.3)
    F, E, S,iter=mf.F_nmf(0.3)
        
			      
