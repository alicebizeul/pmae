"""
Module Name: pca.py
Author: Alice Bizeul
Ownership: ETH Zürich - ETH AI Center
Description: Compute pca and store eigenvalues and principal components
"""

import os
import sys
import glob
import random

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn
from sklearn.decomposition import PCA, IncrementalPCA
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

random.seed(42)

resolution=224
name="imagenet"
data_fn = torchvision.datasets.ImageFolder
folder = "~/ILSVRC2012_img/train"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

trainset = data_fn(root=folder, transform=transform)
num_samples = len(trainset)
subset_size = int(0.1 * num_samples)  # 40% of the dataset
indices = torch.randperm(num_samples)[:subset_size]  # Randomly select indices
subset = Subset(trainset, indices)

# Create a DataLoader for the subset
trainloader = torch.utils.data.DataLoader(subset, batch_size=len(trainset), shuffle=False)
data_iter = iter(trainloader)
images_np, _ = next(data_iter)

images_np = images_np.numpy()
pca_dim=20000
pca = PCA(n_components=pca_dim)  # You can adjust the number of components

# Reshape the images to (num_samples, height * width * channels)
num_samples = images_np.shape[0]
original_shape = images_np.shape
images_np = images_np.reshape(num_samples, -1)

# Standardize
# mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
# images_flat = (images_flat - mean) / std

# Step 4: Perform PCA
pca.fit(images_np)

np.save(f'~/pc_matrix_ipca.npy',pca.components_)
np.save(f'~/eigenvalues_ipca.npy',pca.explained_variance_)
np.save(f'~/eigenvalues_ratio_ipca.npy',pca.explained_variance_ratio_)