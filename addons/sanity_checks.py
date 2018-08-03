import loss_eigenvector as pv_loss
import torch

def assert_similer(exact, ours):
    assert (torch.norm(exact - ours) / torch.norm(exact)) < 1e-4

def test_lossmin(x, Z, loss_min):
    loss_init =  pv_loss.Loss(x, Z)
    print("init/min: %.3f/%.3f" % (loss_init, loss_min))
    print("n: %d, d: %d" % (Z.shape[0], Z.shape[1]))
    assert loss_init > loss_min

def test_gradient(x, Z):
    g_exact = pv_loss.Gradient(x, Z, proj=False, autograd=True)
    g_ours = pv_loss.Gradient(x, Z, proj=False, autograd=False)

    assert_similer(g_exact, g_ours)
    # x.zero_grad()
    g_exact = pv_loss.Gradient(x, Z[0], proj=False, autograd=True)
    g_ours = pv_loss.Gradient(x, Z[0], proj=False, autograd=False)

    print("GRADIENT TEST PASSED...")
    assert_similer(g_exact, g_ours)


def test_batch_loss_grad(x_, Z):
    x = x_.clone() + torch.rand(x_.shape)
    n = float(Z.shape[0])
    F = pv_loss.Loss(x, Z)

    F_sum = 0.
    for i in range(Z.shape[0]):
        F_sum += pv_loss.Loss(x, Z[i])


    assert (abs(F - F_sum/n) / F) < 1e-3

    print("LOSS FUNCTION PASSED...")
    
    G = pv_loss.Gradient(x, Z, proj=True)

    G_sum = G.clone()*0
    for i in range(Z.shape[0]):
        G_sum += pv_loss.Gradient(x, Z[i], proj=True)


    assert_similer(G, G_sum/n)
    print("BATCH GRADIENT (WITH PROJ) PASSED...")
    

    G = pv_loss.Gradient(x, Z, proj=False)

    G_sum = G.clone()*0
    for i in range(Z.shape[0]):
        G_sum += pv_loss.Gradient(x, Z[i], proj=False)


    assert_similer(G, G_sum/n)
    print("BATCH GRADIENT (WITHOUT PROJ) PASSED...")


