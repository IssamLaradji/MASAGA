
import matplotlib
matplotlib.use('Agg')
import math
import torch
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import models as mo
from datasets import datasets as da
from torch.utils import data
import argparse
import torch.nn as nn
import models
from torch.autograd import Variable
import utils as ut
from losses import sphere as sp
from losses import principal_vector_loss as pv_loss
import sys
from addons import sanity_checks

def train(dataset_name, model_name, learning_rate, epochs, 
          sampling_method, project, autograd ,save_img=False):

    history = ut.load_history(dataset_name, model_name, 
                              learning_rate, epochs, sampling_method)
    print("Running {}".format(history["exp_name"]))

    # SET SEED
    np.random.seed(1)
    torch.manual_seed(1) 
    torch.cuda.manual_seed_all(1)

    # Get Datasets
    Z, shape = da.DATASETS[dataset_name]()
    Z = Z - Z.mean(0)
    n = Z.shape[0]

    
    nList = np.arange(n)
    if sampling_method == "uniform":
        P = nList / nList.sum()
    elif sampling_method == "lipschitz":
        L = pv_loss.Lipschitz(Z)
        P = L / L.sum()
       

    if learning_rate == "L":
        L = pv_loss.Lipschitz(Z).max()
        learning_rate = 1./L

    # MODEL
    model = models.MODELS[model_name](Z=Z, 
                        F_func=pv_loss.Loss, 
                        G_func=pv_loss.Gradient, 
                        lr=learning_rate,
                        project=project,
                        autograd=autograd)
    
    # Solve
    x_sol = pv_loss.leading_eigenvecor(Z)
    
    loss_min = pv_loss.Loss(x_sol, Z)

    sanity_checks.test_lossmin(model.x, Z, loss_min)    
    sanity_checks.test_gradient(torch.FloatTensor(x_sol)[:,None], Z)    
    sanity_checks.test_batch_loss_grad(model.x, Z)

    e = 0.
    n_iters = 0.
    i2s = 500
    # Train
    while e < (epochs + 1):
        next_epoch = True

        # inner loop
        for ii in range(n):
            e = n_iters / float(n)
            # Verbose
            if ii % i2s == 0:
                L =  (float((model.F_func(model.x, Z))) - float(loss_min)) / np.abs(float(loss_min))
                
                history["loss"] += [{"loss":L, "epoch":e}]  
                print("Epoch: %.2f/%d - %s - loss: %.3f" % 
                     (e, epochs, history["exp_name"],
                      (L*n)))

            # select a random sample
            i = np.random.choice(nList, replace=True, p=P)

            # make a step
            n_evals = model.step(Z[i], index=i, next_epoch=next_epoch)
            next_epoch = False

            n_iters += n_evals
        
    # After training is done
    ut.save_json(history["path_save"], history)
    import ipdb; ipdb.set_trace()  # breakpoint 33b4f323 //

    torch.save(model.state_dict(), history['path_model'])
    import ipdb; ipdb.set_trace()  # breakpoint 55003f3c //


