#%% 
import os 
import numpy as np
import glob 
import matplotlib.pyplot as plt 
import random 

#%% 
folder ="/cluster/project/sachan/callen/data_alice/"
# tiny 
path_tiny = np.load(folder + "tiny-imagenet-200/train/eigenvalues.npy")
path_tiny_ratio = np.load(folder + "tiny-imagenet-200/train/eigenvalues_ratio.npy")

# blood 
path_blood = np.load(folder + "/medmnist/bloodmnist_eigenvalues.npy")
path_blood_ratio = np.load(folder + "/medmnist/bloodmnist_eigenvalues_ratio.npy")

# derma 
path_derma = np.load(folder + "/medmnist/dermamnist_eigenvalues.npy")
path_derma_ratio = np.load(folder + "/medmnist/dermamnist_eigenvalues_ratio.npy")

# path
path_path = np.load(folder + "/medmnist/pathmnist_eigenvalues.npy")
path_path_ratio = np.load(folder + "/medmnist/pathmnist_eigenvalues_ratio.npy")

# cifar10 
path_cifar10 = np.load(folder + "cifar-10-batches-py/eigenvalues.npy")
path_cifar10_ratio = np.load(folder + "cifar-10-batches-py/eigenvalues_ratio.npy")

# %%

results = {
    "tiny":[],
    "cifar10":[],
    "path":[],
    "blood":[],
    "derma":[],
}
for i in range(1000):
    th_tiny = np.random.randint(1,path_tiny.shape[0],1)[0]
    random.shuffle(path_tiny_ratio)
    results["tiny"].append(sum(path_tiny_ratio[:th_tiny]))

    th_cifar = np.random.randint(1,path_cifar10.shape[0],1)[0]
    random.shuffle(path_cifar10_ratio)
    results["cifar10"].append(sum(path_cifar10_ratio[:th_cifar]))

    th_blood = np.random.randint(1,path_blood.shape[0],1)[0]
    random.shuffle(path_blood_ratio)
    results["blood"].append(sum(path_blood_ratio[:th_blood]))

    th_path = np.random.randint(1,path_path.shape[0],1)[0]
    random.shuffle(path_path_ratio)
    results["path"].append(sum(path_path_ratio[:th_path]))

    th_derma = np.random.randint(1,path_derma.shape[0],1)[0]
    random.shuffle(path_derma_ratio)
    results["derma"].append(sum(path_derma_ratio[:th_derma]))

# Create and save histograms with additional conditions in the title
for dataset, values in results.items():
    # Count the elements lower than 10 and larger than 90
    lower_than_10 = len([x for x in values if x < 0.1])
    larger_than_90 = len([x for x in values if x > 0.9])
    
    # Create the histogram plot
    plt.figure()
    plt.hist(values, bins=30, alpha=0.7, color='blue')
    
    # Add the count information to the title
    plt.title(f'Histogram for {dataset} (Below 10: {lower_than_10}, Above 90: {larger_than_90})')
    plt.xlabel('Sum of eigenvalue ratios')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f"{dataset}_histogram.png")

plt.close('all')
    
# %%
