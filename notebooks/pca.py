#%%

#sbatch -o pca_clevr.out -n 1 --cpus-per-task 4 --mem-per-cpu=80G --time=24:00:00 --wrap="python pca.py"
import sys
sys.path.append("../")
import os 
import glob 
import sklearn
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
# from dataset import TinyImageNetDataset
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from torch.utils.data import DataLoader, Dataset, Subset
import medmnist
from medmnist import DermaMNIST, PathMNIST, BloodMNIST
random.seed(42)
transform = transforms.Compose([
    transforms.ToTensor()
])


data_folder = "/local/home/abizeul/data/"
data_folder = "/cluster/project/sachan/callen/data_alice"

# CIFAR10 # Download and load training dataset
# folder = f'{data_folder}/cifar-10-batches-py'
# trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

# # Fetch the entire dataset in one go
# data_iter = iter(trainloader)
# images, labels = next(data_iter)

# # Step 2: Convert the dataset to NumPy arrays
# images_np = images.numpy()
# labels_np = labels.numpy()

# # Reshape the images to (num_samples, height * width * channels)
# num_samples = images_np.shape[0]
# original_shape = images_np.shape
# images_flat = images_np.reshape(num_samples, -1)

# # Standardize
# mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
# images_flat = (images_flat - mean) / std

# # Step 4: Perform PCA
# pca = PCA()  # You can adjust the number of components
# pca.fit(images_flat)

# np.save(f'{folder}/mean.npy',mean)
# np.save(f'{folder}/std.npy',std)
# np.save(f'{folder}/pc_matrix.npy',pca.components_)
# np.save(f'{folder}/eigenvalues.npy',pca.explained_variance_)
# np.save(f'{folder}/eigenvalues_ratio.npy',pca.explained_variance_ratio_)

# ImageNEt
#%%
# folder = f'{data_folder}/ILSVRC2012_img/train'
# trainset = torchvision.datasets.ImageFolder(folder, transform=transform)
# trainloader =  torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=1)

# #%%
# # Fetch the entire dataset in one go
# data_iter = iter(trainloader)
# images, labels = next(data_iter)
# print("Stage2",flush=True)
#%%
# Step 2: Convert the dataset to NumPy arrays
# images_np = images.numpy()
# labels_np = labels.numpy()
# print("Stage3",flush=True)
# # Reshape the images to (num_samples, height * width * channels)
# num_samples = images_np.shape[0]
# original_shape = images_np.shape
# images_flat = images_np.reshape(num_samples, -1)

# # Standardize
# mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
# images_flat = (images_flat - mean) / std
# print("Stage4",flush=True)
# # # Step 4: Perform PCA
# pca = PCA()  # You can adjust the number of components
# pca.fit(images_flat)
# print("stage 5",flush=True)
# # # save information
# np.save(f'{data_folder}/ILSVRC2012_img/imagenet_mean.npy',mean)
# np.save(f'{data_folder}/ILSVRC2012_img/imagenet_std.npy',std)
# print(pca.components_.shape)
# np.save(f'{data_folder}/ILSVRC2012_img/imagenet_pc_matrix.npy',pca.components_)
# np.save(f'{data_folder}/ILSVRC2012_img/imagenet_eigenvalues.npy',pca.explained_variance_)
# np.save(f'{data_folder}/ILSVRC2012_img/imagenet_eigenvalues_ratio.npy',pca.explained_variance_ratio_)

# TInyimagenet 

# folder = f'{data_folder}/tiny-imagenet-200/train'
# trainset = torchvision.datasets.ImageFolder(folder, transform=transform)
# trainloader =  torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=1)

# # Fetch the entire dataset in one go
# data_iter = iter(trainloader)
# images, labels = next(data_iter)

# # Step 2: Convert the dataset to NumPy arrays
# images_np = images.numpy()
# labels_np = labels.numpy()

# # Reshape the images to (num_samples, height * width * channels)
# num_samples = images_np.shape[0]
# original_shape = images_np.shape
# images_flat = images_np.reshape(num_samples, -1)

# # Standardize
# mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
# images_flat = (images_flat - mean) / std
# nb_components = min(images_flat.shape[0],images_flat.shape[1])

# # Step 4: Perform PCA
# pca = PCA()  # You can adjust the number of components
# pca.fit(images_flat)

# # Step 5: Visualize the result of pc filtering on one image
# def reconstruct_image(pca, components, mean, std_mean, std_stdev, shape):
#     print(mean)
#     image_reconstructed = np.dot(components, pca.components_) + mean
#     image_reconstructed = (image_reconstructed * std_stdev) + std_mean
#     image_reconstructed = image_reconstructed.reshape(shape)
#     return image_reconstructed

# ratios = [0.1,0.2,0.4,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
# nb_plots = len(ratios) + 1
# nb_plots_row = 6
# nb_components = min(images_flat.shape[0],images_flat.shape[1])
# nb_plots_columns = int(np.ceil(nb_plots/nb_plots_row))

# for j, img in enumerate(images_np[:1]):
    # fig, ax = plt.subplots(nb_plots_columns,nb_plots_row,figsize=(20,5))
    # ax[0,0].imshow(np.transpose(img, (1,2,0)),cmap="gray")
    # ax[0,0].set_title("Original Image")
    # ax[0,0].axis('off')

    # fig_, ax_ = plt.subplots(nb_plots_columns,nb_plots_row,figsize=(20,5))
    # ax_[0,0].imshow(np.transpose(img, (1,2,0)),cmap="gray")
    # ax_[0,0].set_title("Original Image")
    # ax_[0,0].axis('off')

    # fig2, ax2 = plt.subplots(nb_plots_columns,nb_plots_row,figsize=(20,5))
    # ax2[0,0].imshow(np.transpose(img, (1,2,0)),cmap="gray")
    # ax2[0,0].set_title("Original Image")
    # ax2[0,0].axis('off')

    # fig2_, ax2_ = plt.subplots(nb_plots_columns,nb_plots_row,figsize=(20,5))
    # ax2_[0,0].imshow(np.transpose(img, (1,2,0)),cmap="gray")
    # ax2_[0,0].set_title("Original Image")
    # ax2_[0,0].axis('off')

    # c=pca.transform(images_flat[j:j+1])
    # cumsum = np.cumsum(pca.explained_variance_ratio_)

    # for i, r in enumerate(ratios):
    #     print("This is ratio",i,r)
    #     row, column = (i+1)//nb_plots_row, (i+1)%nb_plots_row

        # final_component = int(r*nb_components)
        # components = np.zeros_like(c)
        # components[:, :final_component] = c[:, :final_component]
        # reconstructed_image = reconstruct_image(pca, components, pca.mean_, mean, std, [1]+list(original_shape[1:]))

        # ax[row,column].imshow(np.transpose(reconstructed_image[0], (1,2,0)),cmap="gray")
        # ax[row,column].set_title(f"ratio {r}")
        # ax[row,column].axis('off')
        
        #######
        # components = np.zeros_like(c)
        # final_component = nb_components - final_component
        # components[:, -final_component:] = c[:, -final_component:]
        # reconstructed_image = reconstruct_image(pca, components, pca.mean_, mean, std, [1]+list(original_shape[1:]))

        # ax_[row,column].imshow(np.transpose(reconstructed_image[0], (1,2,0)),cmap="gray")
        # ax_[row,column].set_title(f"ratio {round(1-r,2)}")
        # ax_[row,column].axis('off')

        #######
    #     final_component=int(np.argmin(np.abs(cumsum - r)))+1
    #     print("Number of components",final_component)
    #     components = np.zeros_like(c)
    #     components[:, :final_component] = c[:, :final_component]
    #     reconstructed_image = reconstruct_image(pca, components, pca.mean_, mean, std, [1]+list(original_shape[1:]))

    #     ax2[row,column].imshow(np.transpose(reconstructed_image[0], (1,2,0)),cmap="gray")
    #     ax2[row,column].set_title(f"ratio {r}")
    #     ax2[row,column].axis('off')

    #     #######
    #     final_component = nb_components - final_component
    #     components = np.zeros_like(c)
    #     components[:, -final_component:] = c[:, -final_component:]
    #     reconstructed_image = reconstruct_image(pca, components, pca.mean_, mean, std, [1]+list(original_shape[1:]))

    #     ax2_[row,column].imshow(np.transpose(reconstructed_image[0], (1,2,0)),cmap="gray")
    #     ax2_[row,column].set_title(f"ratio {round(1-r,2)}")
    #     ax2_[row,column].axis('off')
    # fig2.savefig('bottom.png') 
    # fig2_.savefig('top.png') 

    # plt.close()

# np.save(f'{folder}/mean.npy',mean)
# np.save(f'{folder}/std.npy',std)
# np.save(f'{folder}/pc_matrix.npy',pca.components_)
# np.save(f'{folder}/eigenvalues.npy',pca.explained_variance_)
# np.save(f'{folder}/eigenvalues_ratio.npy',pca.explained_variance_ratio_)

# DermaMNIST

# folder = f'{data_folder}/medmnist'
# trainset = DermaMNIST(root=folder,split="train",download=True,size=64,transform=transform)
# trainloader =  torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=4)

# # Fetch the entire dataset in one go
# data_iter = iter(trainloader)
# images, labels = next(data_iter)

# # Step 2: Convert the dataset to NumPy arrays
# images_np = images.numpy()
# labels_np = labels.numpy()

# # Reshape the images to (num_samples, height * width * channels)
# num_samples = images_np.shape[0]
# original_shape = images_np.shape
# images_flat = images_np.reshape(num_samples, -1)

# # Standardize
# mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
# images_flat = (images_flat - mean) / std

# # # Step 4: Perform PCA
# pca = PCA()  # You can adjust the number of components
# pca.fit(images_flat)

# # # save information
# np.save(f'{folder}/dermamnist_mean.npy',mean)
# np.save(f'{folder}/dermamnist_std.npy',std)
# print(pca.components_.shape)
# np.save(f'{folder}/derma_pc_matrix.npy',pca.components_)
# np.save(f'{folder}/derma_eigenvalues.npy',pca.explained_variance_)
# np.save(f'{folder}/derma_eigenvalues_ratio.npy',pca.explained_variance_ratio_)

# PAth
# folder = f'{data_folder}/medmnist'
# trainset = PathMNIST(root=folder,split="train",download=True,size=64,transform=transform)
# trainloader =  torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=4)

# # Fetch the entire dataset in one go
# data_iter = iter(trainloader)
# images, labels = next(data_iter)

# # Step 2: Convert the dataset to NumPy arrays
# images_np = images.numpy()
# labels_np = labels.numpy()

# # Reshape the images to (num_samples, height * width * channels)
# num_samples = images_np.shape[0]
# original_shape = images_np.shape
# images_flat = images_np.reshape(num_samples, -1)

# # Standardize
# mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
# images_flat = (images_flat - mean) / std

# # Step 4: Perform PCA
# pca = PCA()  # You can adjust the number of components
# pca.fit(images_flat)

# # save information
# np.save(f'{folder}/pathmnist_mean.npy',mean)
# np.save(f'{folder}/pathmnist_std.npy',std)
# print(pca.components_.shape)
# np.save(f'{folder}/pathmnist_pc_matrix.npy',pca.components_)
# np.save(f'{folder}/pathmnist_eigenvalues.npy',pca.explained_variance_)
# np.save(f'{folder}/pathmnist_eigenvalues_ratio.npy',pca.explained_variance_ratio_)

# Blood 
# folder = f'{data_folder}/medmnist'
# trainset = BloodMNIST(root=folder,split="train",download=True,size=64,transform=transform)
# trainloader =  torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=4)

# # Fetch the entire dataset in one go
# data_iter = iter(trainloader)
# images, labels = next(data_iter)

# # Step 2: Convert the dataset to NumPy arrays
# images_np = images.numpy()
# labels_np = labels.numpy()

# # Reshape the images to (num_samples, height * width * channels)
# num_samples = images_np.shape[0]
# original_shape = images_np.shape
# images_flat = images_np.reshape(num_samples, -1)

# # Standardize
# mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
# images_flat = (images_flat - mean) / std

# # Step 4: Perform PCA
# pca = PCA()  # You can adjust the number of components
# pca.fit(images_flat)

# # save information
# np.save(f'{folder}/bloodmnist_mean.npy',mean)
# np.save(f'{folder}/bloodmnist_std.npy',std)
# print(pca.components_.shape)
# np.save(f'{folder}/bloodmnist_pc_matrix.npy',pca.components_)
# np.save(f'{folder}/bloodmnist_eigenvalues.npy',pca.explained_variance_)
# np.save(f'{folder}/bloodmnist_eigenvalues_ratio.npy',pca.explained_variance_ratio_)

# CLEVR - needs 8 CPUs with 80G each
from torchvision.transforms import v2

folder = f'{data_folder}/CLEVR_v1.0/images/'
img_tf = v2.Compose([
    v2.Resize(size=[224, 224]),
    v2.ToTensor()
])
trainset = torchvision.datasets.ImageFolder(root=folder, transform=img_tf)
trainloader =  torch.utils.data.DataLoader(trainset, batch_size=len(trainset)//20, shuffle=True, num_workers=4)
print("Created the dataloader")
# Fetch the entire dataset in one go
# data_iter = iter(trainloader)
# images, labels = next(data_iter)
mean, std   = torch.tensor(np.load(f'{folder}/clevr_mean.npy')), torch.tensor(np.load(f'{folder}/clevr_std.npy'))

images_np = []
pca = IncrementalPCA()

for i, batch in enumerate(trainloader):
    print(i,len(trainloader),flush=True)
    image, _= batch 
    image = image.reshape([image.shape[0],-1])
    image = (image-mean)/std
    pca.partial_fit(image)

    
    # images_np.append(image.numpy())
# images_np = np.concatenate(images_np,axis=0)
# Step 2: Convert the dataset to NumPy arrays
# images_np = images.numpy()

# Reshape the images to (num_samples, height * width * channels)
# num_samples = images_np.shape[0]
# print("This is the number of samples",num_samples)
# original_shape = images_np.shape
# images_np = images_np.reshape(num_samples, -1)

# Standardize
# mean, std   = np.mean(images_np, axis=0), np.std(images_np, axis=0)
# np.save(f'{folder}/clevr_mean.npy',mean)
# np.save(f'{folder}/clevr_std.npy',std)
# images_np = (images_np - mean) / std
# print("Standardized stuff",flush=True)
# Step 4: Perform PCA
# pca = PCA()  # You can adjust the number of components
print("Did the pca",flush=True)
# save information
print(pca.components_.shape,flush=True)
np.save(f'{folder}/clevr_pc_matrix_large_large.npy',pca.components_)
np.save(f'{folder}/clevr_eigenvalues_large_large.npy',pca.explained_variance_)
np.save(f'{folder}/clevr_eigenvalues_ratio_large_large.npy',pca.explained_variance_ratio_)