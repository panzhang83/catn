import numpy as np
import torch
import time

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A, Dnew=-1):
        #t0 = time.time()
        U, S, V = torch.svd(A)
        #print(time.time() - t0)
        if( Dnew < 0 ):
            Dnew = min(A.shape[0],A.shape[1])
        
        self.save_for_backward(U[:, :Dnew], S[:Dnew], V[:, :Dnew])
        return U[:, :Dnew], S[:Dnew], V[:, :Dnew]

    @staticmethod
    def backward(self, dU, dS, dV, epsilon=1E-50):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)
        S2 = S**2
        Sinv = S/(S2 + epsilon)
        Finv = S2 - S2[:,None]
        F = Finv/(Finv**2 + epsilon)

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = F * (UdU-UdU.t()) * S
        Sv = S[:,None] * (F*(VdV-VdV.t()))

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU*Sinv) @ Vt 
        if (N>NS):
            dA = dA + (U*Sinv) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        #print (dU.norm().item(), dS.norm().item(), dV.norm().item())
        #print (Su.norm().item(), Sv.norm().item(), dS.norm().item())
        #print (dA1.norm().item(), dA2.norm().item(), dA3.norm().item())
        return dA

def test_svd():
    M, N = 50, 20
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVD.apply, (input,), eps=1e-6, atol=1e-8))
    print("Test Pass!")

def beachmark_svd():
    M, N = 10, 10
    print('col / row of tensor:', M, N)
    torch.manual_seed(2)
    svd = SVD.apply
    A = torch.rand(M,N, dtype=torch.float64, device='cpu')
    A_cuda = torch.rand(M,N, dtype=torch.float64, device='cuda:7')
    t0 = time.time()
    U, S, V = svd(A)
    print('cpu_svd_time ', time.time() - t0)
    t0 = time.time()
    U, S, V = svd(A_cuda)
    print('gpu_svd_time ', time.time() - t0)

if __name__=='__main__':
    beachmark_svd()
