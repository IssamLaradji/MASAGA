import torch

import numpy as np
from core import sphere
import utils as ut 

#------- LEADING EIGENVECTOR OBJECTIVE
def reshape(x, Z):
    if isinstance(x, np.ndarray):
        x = torch.FloatTensor(x)

    if x.dim() == 1:
        x = x.unsqueeze(1)        
    if Z.dim() == 1:
        Z = Z.unsqueeze(0)

    return x, Z

def Loss(x, Z):
    x, Z = reshape(x, Z)

    A = Z.t().mm(Z)
    n = float(Z.shape[0])

    f = - x.t().mm(A.mm(x))
    return f.squeeze() / n

def Gradient(x, Z, proj=True, autograd=False):
    x, Z = reshape(x, Z)
    n = float(Z.shape[0])

    if autograd:        
        G = ut.compute_grad(Loss, x, Z)
    else:
        A = Z.t().mm(Z)
        G = - 2. * A.mm(x) / n
        
    if proj:
        return sphere.Proj(x, G)
    else:
        return G

def Lipschitz(Z):
    n = float(Z.shape[0])    
    L = np.zeros(int(n))

    for i in range(int(n)):
        L[i] = (Z[i]**2).sum().item()

    return L

## LEADING EIGEN
def leading_eigenvecor(Z):
    Z = np.asarray(Z)
    np.random.seed(1)
    eigh = np.linalg.eigh(Z.T.dot(Z))

    #assert False not in (eigh[0]>0)
    return eigh[1][:, -1]
