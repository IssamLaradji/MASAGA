import torch

# Riemannian sphere operations
def Exp(x, U):
    U_norm = torch.norm(U)
    
    A = torch.cos(U_norm) * x
    B = torch.sin(U_norm) * U / U_norm

    return A + B

def Retract(A, B):
    return (A + B) / float(torch.norm(A + B))


def Transport(U, x, y):
    return Proj(y, U)

def Proj(x, H):
    return H - x.t().mm(H) * x
    