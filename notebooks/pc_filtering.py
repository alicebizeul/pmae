## python notebooks/pc_filtering.py  --ratio 0.99 --existing_eigen /local/home/abizeul/data/tinyimagenet_0.99_percent/pc_train/pc_train.joblib --split "val"
## python notebooks/pc_filtering.py  --ratio 0.99  --split "train"

# %%
import torch
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
import os 
import torchvision
from torchvision import transforms, models
from collections import defaultdict
from jax import vmap
from jax import jit
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
from plotly.subplots import make_subplots
import argparse
import random
import pickle
import joblib


# %%
@jit
def euclidean_distances_jax(matrix1, matrix2):
    # Compute the pairwise Euclidean distances between rows of matrix1 and matrix2
    diff = matrix1[:, jnp.newaxis, :] - matrix2[jnp.newaxis, :, :]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    return dist

# %%
def reconstruct_image(pca, components, mean, shape):
    image_reconstructed = np.dot(components, pca.components_) + mean
    image_reconstructed = image_reconstructed.reshape(shape)
    return image_reconstructed



# %%
def main(args):
    # %% [markdown]
    # ## TinyImageNet

    if 'args' in vars():
        dataset = args.dataset
        ratio = args.ratio
        savedir = args.savedir
        if savedir is None:
            savedir = f"/local/home/abizeul/data/{dataset}_{ratio}_percent/{args.split}"
            savedir_eigen = f"/local/home/abizeul/data/{dataset}_{ratio}_percent/pc_{args.split}"
    else:
        dataset = "tinyimagenet"
        ratio = 0.5
        savedir = f"/local/home/abizeul/data/{dataset}_{ratio}_percent/{args.split}"
        savedir_eigen = f"/local/home/abizeul/data/{dataset}_{ratio}_percent/pc_{args.split}"

    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    if not os.path.isdir(savedir_eigen):
        os.makedirs(savedir_eigen)

    transform = transforms.ToTensor()
    if dataset == "tinyimagenet":
        trainset = torchvision.datasets.ImageFolder(f'/local/home/abizeul/data/tiny-imagenet-200/{args.split}', transform=transform)        
    else: 
        raise NotImplemented

    # %%
    seed=1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # %%
    # PCA
    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=4)

    # Fetch the entire dataset in one go
    data_iter      = iter(trainloader)
    images, labels = next(data_iter)

    # Step 2: Convert the dataset to NumPy arrays
    images_np = images.numpy()
    labels_np = labels.numpy()

    # Reshape the images to (num_samples, height * width * channels)
    num_samples    = images_np.shape[0]
    original_shape = images_np.shape

    images_flat    = images_np.reshape(num_samples, -1)
    num_features   = images_flat.shape[-1]
    num_components = min(num_samples,num_features)

    # Step 3: standardize
    mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
    images_flat = (images_flat - mean) / std

    # Subsample
    # sub_samples = 1500
    # indexes     = np.random.choice(images_flat.shape[0], sub_samples, replace=False)
    # images_flat = images_flat[indexes,:]
    # labels_np   = np.expand_dims(np.array(labels_np),-1)
    # labels_np   = labels_np[indexes]

    # mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
    # images_flat = ((images_flat - mean)/std)
    # mean, std   = np.mean(labels_np, axis=0), np.std(labels_np, axis=0)
    # labels_np   = ((labels_np - mean)/std)

    # Step 4: Perform PCA
    if args.existing_eigen is not None:
        pca = joblib.load(args.existing_eigen)
    else:
        pca = PCA()  # You can adjust the number of components
        pca.fit(images_flat)
        joblib.dump(pca,os.path.join(savedir_eigen,f"pc_{args.split}.joblib"))

    c=pca.transform(images_flat)

    # ratio 
    print(ratio,num_components,int(ratio*num_components))
    i = int(ratio * num_components)
    components = np.zeros_like(c)
    components[:, -i:] = c[:, -i:]
    reconstructions=reconstruct_image(pca, components, pca.mean_, [c.shape[0]]+list(original_shape[1:]))
    reconstructions=((reconstructions.reshape(num_samples, -1)*std)+mean).reshape(reconstructions.shape)
    for index, (recon,label) in enumerate(zip(reconstructions,labels_np)):
        print(label)
        if not os.path.isdir(f"{savedir}/{label}"): 
            os.makedirs(f"{savedir}/{label}")
        torchvision.utils.save_image(torch.tensor(recon), f'{savedir}/{label}/{index}_{label}.png')

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A more complex script example.")
    
    # Required positional argument
    parser.add_argument('--dataset', type=str,   default="tinyimagenet", help='The file to process.')
    parser.add_argument('--dim',     type=int,   default=512,            help='Number of times to process the file.')
    parser.add_argument('--ratio',   type=float, default=0.5,            help='Number of times to process the file.')
    parser.add_argument('--savedir', type=str,   default=None,           help='Number of times to process the file.')
    parser.add_argument('--split',   type=str,   default="train",        help='Number of times to process the file.')
    parser.add_argument('--existing_eigen',   type=str,   default=None,        help='Number of times to process the file.')

    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
