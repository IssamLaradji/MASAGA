import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import copy

from losses import sphere as sp


class Base(nn.Module):
    def __init__(self, Z, F_func, G_func, lr):
        super().__init__()
        self.F_func = F_func 
        self.G_func = G_func
        self.lr = lr
        
        self.n = Z.shape[0]
        d = Z.shape[1]
        self.Z = Z
        
        # Initialize x
        np.random.seed(1)
        #r = np.random.randn(Z.shape[1], 1)
        r = torch.ones(d, 1)
        self.x  = nn.Parameter(r / np.sqrt(d))

        self.proj = True
        self.autograd = False

#--------------------------------------------------
class SVRG(Base):
    def __init__(self, Z, F_func, G_func, lr):
        super().__init__(Z, F_func, G_func, lr)
        
        self.x_outer = self.x.clone().detach()
        self.mu = torch.zeros(self.x.shape)

    @torch.no_grad()
    def step(self, Zi, **extras):
        n_evals = 2

        # GET FULL MEAN IF NEEDED
        if extras["next_epoch"] is True:
            self.mu = self.G_func(self.x, self.Z, proj=self.proj, 
                                  autograd=self.autograd)

            self.x_outer = self.x.clone().detach()
            n_evals += self.Z.shape[0]

        g_inner = self.G_func(self.x, Zi, proj=self.proj, autograd=self.autograd)
        g_outer = self.G_func(self.x_outer, Zi, proj=self.proj, autograd=self.autograd)

        V = g_inner - sp.Transport(g_outer - self.mu, None, self.x)
        self.x.data = sp.Exp(self.x, -self.lr * V)

        return n_evals


#--------------------------------------------------
class SGD(Base):
    def __init__(self, Z, F_func, G_func, lr):
        super().__init__(Z, F_func, G_func, lr)
    
    @torch.no_grad()
    def step(self, Zi, **extras):
        n_evals = 1

        g = self.G_func(self.x, Zi, proj=self.proj, autograd=self.autograd)

        self.x.data = sp.Exp(self.x, -self.lr * g)

        return n_evals

#--------------------------------------------------
class SAGA(Base):
    def __init__(self, Z, F_func, G_func, lr):
        super().__init__(Z, F_func, G_func, lr)

        self.x_init = self.x.clone().detach()

        shape = self.x.shape
        self.mu = torch.zeros(shape)
        self.M = []
        for i in range(self.n):
            self.M += [torch.zeros(shape)]
    
    @torch.no_grad()
    def step(self, Zi, **extras):
        n_evals = 1
        index = extras["index"]
        n = self.Z.shape[0]

        Mi = self.M[index]

        g = self.G_func(self.x, Zi, 
                        proj=self.proj, 
                        autograd=self.autograd)

        V = g - sp.Transport(Mi - self.mu, None, self.x)
        self.x.data = sp.Exp(self.x, -self.lr * V)
        
        # Update previous grad and mean
        g = sp.Transport(g, None, self.x_init)
        self.mu += (1./n) * (g - Mi)
        self.M[index] = g.clone()


        return n_evals


MODELS = {"svrg":SVRG, "sgd":SGD, "saga":SAGA}
