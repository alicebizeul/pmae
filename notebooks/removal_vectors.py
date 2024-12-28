#%% 

import numpy as np 
import os 
import matplotlib.pyplot as plt 


folder = "/cluster/project/sachan/callen/data_alice"
derma = np.load(folder+"/medmnist/dermamnist_eigenvalues_ratio.npy")
blood = np.load(folder+"/medmnist/bloodmnist_eigenvalues_ratio.npy")
path = np.load(folder+"/medmnist/pathmnist_eigenvalues_ratio.npy")
tiny = np.load(folder+"/tiny-imagenet-200/train/eigenvalues_ratio.npy")
cifar10 = np.load(folder+"/cifar-10-batches-py/eigenvalues_ratio.npy")
# %%
dim_derma, dim_blood, dim_path, dim_tiny, dim_cifar = 3*64*64,  3*64*64,  3*64*64,  3*64*64,  3*32*32 
samples_derma, samples_blood, samples_path, samples_tiny, samples_cifar = 7000, 12000, 90000, 100000, 50000


# %%
num_low_eigen = lambda x: round(sum(1*(x>0.01))/x.shape[0],4)

num_derma = num_low_eigen(derma)
num_path = num_low_eigen(path)
num_blood = num_low_eigen(blood)
num_tiny = num_low_eigen(tiny)
num_cifar = num_low_eigen(cifar10)
print(num_derma,num_path,num_blood,num_cifar,num_tiny)
# %%
# %%
num_low_eigen = lambda x: round(sum(1*(x>0.0001))/x.shape[0],4)

num_derma = num_low_eigen(derma)
num_path = num_low_eigen(path)
num_blood = num_low_eigen(blood)
num_tiny = num_low_eigen(tiny)
num_cifar = num_low_eigen(cifar10)
print(num_derma,num_path,num_blood,num_cifar,num_tiny)
# %%
