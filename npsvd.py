import numpy as np
import time
import sys
from scipy import linalg

def svd(A):
    try:
        U,s,V = np.linalg.svd(A)
    except:
        U,s,V = linalg.svd(A,lapack_driver='gesvd')
    return U,s,V.T


def rsvd(A,k=100,oversample=10, power=10, ortho=False):
    sys.stdout.write("R");sys.stdout.flush()
    m,n=A.shape
    p = min(n,oversample*k)
    Y = A@ (np.random.randn(n,p).astype(A.dtype))

    for i in range(power):
        if(ortho):
            # method 1
            #[Q,R] = np.linalg.qr(Y)
            #[Q,R] = np.linalg.qr(A.T@Q)
            # method 2
            #[Q,R] = np.linalg.qr(Y)
            #Y = A.t()@Q
            # method 3
            [Y,_] = np.linalg.qr(A.T@Y)
        else:
            Y = A.T@Y
        Y = A@Y
    [Q,R] = np.linalg.qr(Y)
    B = Q.T@A

    [U,s,V] = np.linalg.svd(B)
    V = V.T
    U = Q@U
    k = min(k,U.shape[1])
    U = U[:,:k]
    V = V[:,:k]
    s = s[:k]
    return U,s,V


