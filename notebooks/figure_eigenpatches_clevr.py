
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
from sklearn.decomposition import PCA
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
    transforms.Resize(size=[224, 224]),
    transforms.ToTensor()
])
#sbatch -o pca_clevr.out -n 1 --cpus-per-task 4 --mem-per-cpu=80G --time=24:00:00 --wrap="python pca.py"
#sbatch -o eigenpatch.out -n 1 --cpus-per-task 4 --mem-per-cpu=80G --time=1:00:00 --wrap="python figure_eigenpatches.py"

# save information
data_folder = "/cluster/project/sachan/callen/data_alice"
folder = f'{data_folder}/CLEVR_v1.0/images/'

def reconstruct_image(pca, components,std_mean, std_stdev, shape):
    image_reconstructed = np.dot(components, pca)
    image_reconstructed = image_reconstructed.reshape(shape)
    image_reconstructed = (image_reconstructed * std_stdev) + std_mean
    return image_reconstructed

trainset = torchvision.datasets.ImageFolder(folder, transform=transform)
trainloader =  torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)
data_iter = iter(trainloader)
images, labels = next(data_iter)

c=np.load(f'{folder}/clevr_pc_matrix.npy').T
# eigenvalues=np.load(f'{folder}/eigenvalues.npy')
ratio=np.load(f'{folder}/clevr_eigenvalues_ratio.npy')


original_shape = images.shape
mean = np.reshape(np.load(f'{folder}/clevr_mean.npy'),original_shape[1:])
std = np.reshape(np.load(f'{folder}/clevr_std.npy'),original_shape[1:])
images = (images - mean)/std
images = images.reshape(images.shape[0],-1)
projection = images @ c

fig, ax = plt.subplots(2,2,figsize=(20,5))

original= (images.reshape(original_shape)* std) + mean
row=0
column=0
ax[row,column].imshow(np.transpose(original[0], (1,2,0)),cmap="gray")
ax[row,column].axis('off')


r_start, r_stop, nb_components = 0, 0.01, ratio.shape[0]
start_component = int(r_start*nb_components)
final_component = int(r_stop*nb_components)

components = np.zeros_like(projection)
components[:, start_component:final_component] = projection[:, start_component:final_component]
reconstructed_image = reconstruct_image(c.T, components, mean, std, original_shape)

row=1
column=0
ax[row,column].imshow(np.transpose(reconstructed_image[0], (1,2,0)),cmap="gray")
ax[row,column].axis('off')

r_start, r_stop, nb_components = 0.01, 0.05, ratio.shape[0]
start_component = int(r_start*nb_components)
final_component = int(r_stop*nb_components)

components = np.zeros_like(projection)
components[:, start_component:final_component] = projection[:, start_component:final_component]
reconstructed_image = reconstruct_image(c.T, components, mean, std, original_shape)

row=0
column=1
ax[row,column].imshow(np.transpose(reconstructed_image[0], (1,2,0)),cmap="gray")
ax[row,column].axis('off')

find_threshold = lambda myeigenvalues ,ratio: np.argmin(np.abs(np.cumsum(myeigenvalues) - ratio))
start_component, stop_component, nb_components = find_threshold(ratio,0.90), find_threshold(ratio,1.0), ratio.shape[0]
print(start_component,stop_component,ratio[start_component])

components = np.zeros_like(projection)
components[:, start_component:final_component] = projection[:, start_component:final_component]
reconstructed_image = reconstruct_image(c.T, components, mean, std, original_shape)

row=1
column=1
ax[row,column].imshow(np.transpose(reconstructed_image[0], (1,2,0)),cmap="gray")
ax[row,column].axis('off')
fig.savefig(f"/cluster/home/callen/projects/mae/notebooks/eigenvalues_patch2.png")

print(c.shape,ratio.shape)
c = c[:,:start_component].T
ratio = ratio[:start_component]
print(c.shape,ratio.shape)
np.save(f"{folder}/clevr_eigenvalues_ratio_reduced.npy",ratio)
np.save(f"{folder}/clevr_pc_matrix_reduced.npy",c)
