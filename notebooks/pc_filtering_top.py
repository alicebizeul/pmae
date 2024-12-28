## python notebooks/pc_filtering.py  --ratio 0.99 --existing_eigen /local/home/abizeul/data/tinyimagenet_0.99_percent/pc_train/pc_train.joblib --existing_mean /local/home/abizeul/data/tinyimagenet_0.99_percent/pc_train/mean.npy --existing_std /local/home/abizeul/data/tinyimagenet_0.99_percent/pc_train/std.npy --split "val"
## python notebooks/pc_filtering.py  --ratio 0.99  --split "train"


## python notebooks/pc_filtering_top.py  --ratio 0.99 --existing_eigen /local/home/abizeul/data/tinyimagenet_0.99_percent/pc_train/pc_train.joblib --existing_mean /local/home/abizeul/data/tinyimagenet_0.99_percent/pc_train/mean.npy --existing_std /local/home/abizeul/data/tinyimagenet_0.99_percent/pc_train/std.npy --split "val"
## python notebooks/pc_filtering_top.py  --ratio 0.99  --split "train" --root /cluster/home/abizeul/mae/outputs --datadir /cluster/work/vogtlab/Group/abizeul/data/tiny-imagenet-200/ --dataset tinyimagenet

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt 
import os 
import torchvision
from torchvision import transforms, models
from collections import defaultdict
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
from plotly.subplots import make_subplots
import argparse
import random
import pickle
import joblib
import plotly.graph_objects as go




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
        datafolder = args.datadir
        if savedir is None:
            savedir_top = f"{args.root}/{dataset}_{ratio}_percent_top/{args.split}"
            savedir_bottom = f"{args.root}/{dataset}_{ratio}_percent_bottom/{args.split}"
            savedir_eigen = f"{args.root}/{dataset}_{ratio}_percent_top/pc_{args.split}"
    else:
        dataset = "tinyimagenet"
        ratio = 0.5
        datafolder = "/local/home/abizeul/data/tiny-imagenet-200"
        savedir_top = f"/local/home/abizeul/data/{dataset}_{ratio}_percent_top/{args.split}"
        savedir_bottom = f"/local/home/abizeul/data/{dataset}_{ratio}_percent_bottom/{args.split}"
        savedir_eigen = f"/local/home/abizeul/data/{dataset}_{ratio}_percent_top/pc_{args.split}"

    if not os.path.isdir(savedir_top):
        os.makedirs(savedir_top)
    if not os.path.isdir(savedir_bottom):
        os.makedirs(savedir_bottom)
    if not os.path.isdir(savedir_eigen):
        os.makedirs(savedir_eigen)

    transform = transforms.ToTensor()
    if dataset == "tinyimagenet":
        trainset = torchvision.datasets.ImageFolder(f'{datafolder}/{args.split}', transform=transform)        
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
    i = int(ratio * num_components)
    j = num_components-i

    # Step 3: standardize


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
        mean = np.load(args.existing_mean)
        std = np.load(args.existing_std)
        images_flat = (images_flat - mean) / std
    else:
        mean, std   = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
        images_flat = (images_flat - mean) / std
        pca = PCA()  # You can adjust the number of components
        pca.fit(images_flat)
        eigenvalues = pca.explained_variance_
        print("The amount of information",sum(pca.explained_variance_[:j]),sum(pca.explained_variance_[-i:]))

        x = np.arange(eigenvalues.shape[0])
        y = list(eigenvalues)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=x, 
            y=y,
            marker=dict(color='rgba(58, 71, 80, 0.6)', line=dict(color='rgba(58, 71, 80, 1.0)', width=1.5)),
            opacity=0.8
        ))

        # Update layout for aesthetics
        fig.update_layout(
            title='Vertical Bar Plot',
            xaxis=dict(
                title='Labels',
                titlefont=dict(size=14),
                tickfont=dict(size=12),
                showgrid=True,  # Add grid lines for y-axis
                gridcolor='lightgrey',  # Color of grid lines
                gridwidth=0.5  # Width of grid lines
            ),
            yaxis=dict(
                title='Values (Log Scale)',
                titlefont=dict(size=14),
                tickfont=dict(size=12),
                type='log',
                showgrid=True,  # Add grid lines for y-axis
                gridcolor='lightgrey',  # Color of grid lines
                gridwidth=0.5  # Width of grid lines
            ),
            plot_bgcolor='white',
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Show the plot
        fig.write_image(os.path.join(savedir_eigen,f"pc_filtering_images_{args.split}.png"))

        joblib.dump(pca,os.path.join(savedir_eigen,f"pc_{args.split}.joblib"))
        np.save(os.path.join(savedir_eigen,"mean.npy"),mean)
        np.save(os.path.join(savedir_eigen,"std.npy"),std)

    c=pca.transform(images_flat)

    # top 
    print(ratio,num_components,int(ratio*num_components))
    components = np.zeros_like(c)
    components[:, :j] = c[:, :j]
    reconstructions=reconstruct_image(pca, components, pca.mean_, [c.shape[0]]+list(original_shape[1:]))
    reconstructions=((reconstructions.reshape(num_samples, -1)*std)+mean).reshape(reconstructions.shape)
    for index, (recon,label) in enumerate(zip(reconstructions,labels_np)):
        if not os.path.isdir(f"{savedir_top}/{label}"): 
            os.makedirs(f"{savedir_top}/{label}")
        torchvision.utils.save_image(torch.tensor(recon), f'{savedir_top}/{label}/{index}_{label}.png')


    # bottom
    components = np.zeros_like(c)
    components[:, -i:] = c[:, -i:]
    reconstructions=reconstruct_image(pca, components, pca.mean_, [c.shape[0]]+list(original_shape[1:]))
    reconstructions=((reconstructions.reshape(num_samples, -1)*std)+mean).reshape(reconstructions.shape)
    for index, (recon,label) in enumerate(zip(reconstructions,labels_np)):
        if not os.path.isdir(f"{savedir_bottom}/{label}"): 
            os.makedirs(f"{savedir_bottom}/{label}")
        torchvision.utils.save_image(torch.tensor(recon), f'{savedir_bottom}/{label}/{index}_{label}.png')

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A more complex script example.")
    
    # Required positional argument
    parser.add_argument('--root',    type=str,   default="/local/home/abizeul", help='The file to process.')
    parser.add_argument('--dataset', type=str,   default="tinyimagenet", help='The file to process.')
    parser.add_argument('--dim',     type=int,   default=512,            help='Number of times to process the file.')
    parser.add_argument('--ratio',   type=float, default=0.5,            help='Number of times to process the file.')
    parser.add_argument('--savedir', type=str,   default=None,           help='Number of times to process the file.')
    parser.add_argument('--datadir', type=str,   default=None,           help='Number of times to process the file.')
    parser.add_argument('--split',   type=str,   default="train",        help='Number of times to process the file.')
    parser.add_argument('--existing_eigen',   type=str,   default=None,        help='Number of times to process the file.')
    parser.add_argument('--existing_mean',    type=str,   default=None,        help='Number of times to process the file.')
    parser.add_argument('--existing_std',     type=str,   default=None,        help='Number of times to process the file.')

    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
