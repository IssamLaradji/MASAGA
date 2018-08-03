from torch.utils import data
import numpy as np
import torch

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

# from scipy.misc import imread
# from scipy.misc import imresize

from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
# from skimage.io import imread
# from skimage.transform import rescale
import glob

def ocean():    
    ls = glob.glob("datasets/ocean/*.jpg")

    Z = []

    #shape = (100,100)

    shape = zoom(imread(ls[0]),0.3).shape[:2]

    for l in ls:
        Z += [zoom(imread(l),0.3).ravel()]
    Z = np.array(Z) / 255.
    
    return torch.FloatTensor(Z), shape

def mnist():
    # SET SEED
    np.random.seed(1)
    torch.manual_seed(1) 
    torch.cuda.manual_seed_all(1)
    mnist = fetch_mldata('MNIST original', data_home="/mnt/home/issam/")

    X, y = shuffle(mnist["data"][:60000], mnist["target"][:60000])
    X = X / 255.
    y = y

    # X = X[:10000]

    n, d = X.shape
    Z = X
    Z = torch.FloatTensor(X)

    return Z, (28,28)

def M04():
    # SET SEED
    np.random.seed(1)
    torch.manual_seed(1) 
    torch.cuda.manual_seed_all(1)
    try:
        mnist = fetch_mldata('MNIST original', data_home="/mnt/home/issam/Datasets/")
    except:
        mnist = fetch_mldata('MNIST original')
    X, y = shuffle(mnist["data"], mnist["target"])
    X = X / 255.
    y = y

    ind = y <= 4
    X = X[ind][:2000]

    n, d = X.shape
    Z = X
    Z = torch.FloatTensor(X)

    return Z

def M59():
    # SET SEED
    np.random.seed(1)
    torch.manual_seed(1) 
    torch.cuda.manual_seed_all(1)
    try:
        mnist = fetch_mldata('MNIST original', data_home="/mnt/home/issam/Datasets/")
    except:
        mnist = fetch_mldata('MNIST original')
    X, y = shuffle(mnist["data"], mnist["target"])
    X = X / 255.
    y = y

    ind = y >= 5
    X = X[ind][:2000]

    n, d = X.shape
    Z = X
    Z = torch.FloatTensor(X)

    return Z

def A():
    # SET SEED
    np.random.seed(1)
    torch.manual_seed(1) 
    torch.cuda.manual_seed_all(1)


    n, d = 1000, 100
    Z = np.random.randint(1,100,n)[:,None]*np.random.rand(n, d) 
    

    Z = torch.FloatTensor(Z)

    return Z, None

def B():
    # SET SEED
    np.random.seed(1)
    torch.manual_seed(1) 
    torch.cuda.manual_seed_all(1)


    n, d = 10000, 500
    Z = np.random.randint(1,100,n)[:,None]*np.random.rand(n, d) 
    

    Z = torch.FloatTensor(Z)

    return Z

# def C():
#     # SET SEED
#     np.random.seed(1)
#     torch.manual_seed(1) 
#     torch.cuda.manual_seed_all(1)


#     n, d = 10000, 1000
#     Z = np.random.rand(n, d)
#     Z = torch.FloatTensor(Z)

#     return Z


# def B():
#     # SET SEED
#     np.random.seed(1)
#     torch.manual_seed(1) 
#     torch.cuda.manual_seed_all(1)


#     n, d = 10000, 1000
#     Z = np.random.rand(n, d)
#     Z = torch.FloatTensor(Z)

#     return Z

def M(label):
    # SET SEED
    np.random.seed(1)
    torch.manual_seed(1) 
    torch.cuda.manual_seed_all(1)
    try:
        mnist = fetch_mldata('MNIST original', data_home="/mnt/home/issam/Datasets/")
    except:
        mnist = fetch_mldata('MNIST original')
    X, y = shuffle(mnist["data"], mnist["target"])
    X = X / 255.
    y = y

    ind = y == label
    X = X[ind][:500]

    n, d = X.shape
    Z = X
    Z = torch.FloatTensor(X)

    return Z, (28,28)

# def B():
#     # SET SEED
#     np.random.seed(1)
#     torch.manual_seed(1) 
#     torch.cuda.manual_seed_all(1)


#     # DATASET - min loss 2.190
#     n, d = 100, 10
#     Z = torch.FloatTensor(np.random.rand(n, d))

#     r = np.random.rand(d,1)
#     x = torch.FloatTensor(r / np.linalg.norm(r))

#     return x, Z

#Dataset
class synthetic(data.Dataset):
    def __init__(self):
        bias = 1; scaling = 10; 
        sparsity = 10; solutionSparsity = 0.1;
        n = 100;
        p = 10;
        A = np.random.randn(n,p)+bias;
        A = A.dot(np.diag(scaling* np.random.randn(p)))
        A = A * (np.random.rand(n,p) < (sparsity*np.log(n)/n));
        w = np.random.randn(p) * (np.random.rand(p) < solutionSparsity);

        b = A.dot(w)


        self.X = A

        self.y = b[:, np.newaxis]
        self.n = A.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        Xi = torch.FloatTensor(self.X[index])
        yi = torch.FloatTensor(self.y[index])

        return {"X":Xi, 
                "y":yi,
                "ind":index}




DATASETS = {"synthetic":A, "Mnist":mnist, "ocean":ocean, 
            "M1":lambda : M(label=1),
            "M2":lambda : M(label=2),
            "M3":lambda : M(label=3),
            "M4":lambda : M(label=4),
            "M5":lambda : M(label=5),
            "M6":lambda : M(label=6),
            "M7":lambda : M(label=7)}

