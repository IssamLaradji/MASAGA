import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
from datasets import datasets as da
import models
import utils as ut
import loss_eigenvector
from addons import sanity_checks

def get_learning_rate(L, string):
    return 1. / (L*int(string[1:]))

def train(dataset_name, model_name, learning_rate, epochs, 
          sampling_method, reset, save_img=False):

    history = ut.load_history(dataset_name, model_name, 
                              learning_rate, epochs, 
                              sampling_method, reset)
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
        L = loss_eigenvector.Lipschitz(Z)
        P = L / L.sum()
    
    L = loss_eigenvector.Lipschitz(Z).max()    
    lr = get_learning_rate(L, learning_rate)

    # MODEL
    model = models.MODELS[model_name](Z=Z, 
                        F_func=loss_eigenvector.Loss, 
                        G_func=loss_eigenvector.Gradient, 
                        lr=lr)
    # Solve    
    x_sol = loss_eigenvector.leading_eigenvecor(Z)
    loss_min = loss_eigenvector.Loss(x_sol, Z)
    # sanity_checks.test_lossmin(model.x, Z, loss_min)    
    # sanity_checks.test_gradient(torch.FloatTensor(x_sol)[:,None], Z)    
    # sanity_checks.test_batch_loss_grad(model.x, Z)
    e = 0.
    n_iters = 0.
    # Train
    while e < (epochs + 1):
        next_epoch = True

        # inner loop
        for ii in range(n):
            e = n_iters / float(n)
            # Verbose
            if ii % 500 == 0:
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
    torch.save(model.state_dict(), history['path_model'])
    print("model saved in {}".format( history['path_model']))

