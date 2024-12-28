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
import random

# %%
@jit
def euclidean_distances_jax(matrix1, matrix2):
    # Compute the pairwise Euclidean distances between rows of matrix1 and matrix2
    diff = matrix1[:, jnp.newaxis, :] - matrix2[jnp.newaxis, :, :]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    return dist

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
        dim = args.dim
    else:
        dataset = "tinyimagenet"
        dim = 512
    transform = transforms.ToTensor()

    if dataset == "tinyimagenet":
        trainset = torchvision.datasets.ImageFolder('/local/home/abizeul/data/tiny-imagenet-200/train', transform=transform)        
        testset = torchvision.datasets.ImageFolder('/local/home/abizeul/data/tiny-imagenet-200/train', transform=transform)        
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
    # concatenate
    # for k in list(label_dict.keys()):
    #     label_dict[k] = np.concatenate(label_dict[k],axis=0)

    # %%
    # PCA
    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=4)

    # Fetch the entire dataset in one go
    data_iter = iter(trainloader)
    images, labels = next(data_iter)

    # Step 2: Convert the dataset to NumPy arrays
    images_np = images.numpy()
    labels_np = labels.numpy()

    # Reshape the images to (num_samples, height * width * channels)
    num_samples = images_np.shape[0]
    original_shape = images_np.shape

    images_flat = images_np.reshape(num_samples, -1)
    num_features = images_flat.shape[-1]
    num_components = min(num_samples,num_features)

    # Step 3: standardize
    mean, std = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
    images_flat = (images_flat - mean) / std

    # Subsample
    sub_samples = 15000
    indexes = np.random.choice(images_flat.shape[0], sub_samples, replace=False)
    images_flat = images_flat[indexes,:]
    labels_np = np.expand_dims(np.array(labels_np),-1)
    labels_np = labels_np[indexes]

    mean, std = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
    images_flat = ((images_flat - mean)/std)
    mean, std = np.mean(labels_np, axis=0), np.std(labels_np, axis=0)
    labels_np = ((labels_np - mean)/std)

    # Step 4: Perform PCA
    pca = PCA()  # You can adjust the number of components
    pca.fit(images_flat)
    c=pca.transform(images_flat)

# %%
# Eval data
trainset = torchvision.datasets.ImageFolder('/local/home/abizeul/data/tiny-imagenet-200/train', transform=transform)
trainloader =  torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=4)

# Fetch the entire dataset in one go
data_iter = iter(trainloader)
images, labels = next(data_iter)

# Step 2: Convert the dataset to NumPy arrays
images_np = images.numpy()
labels_np = labels.numpy()

# Reshape the images to (num_samples, height * width * channels)
num_samples = images_np.shape[0]
original_shape = images_np.shape
images_flat = images_np.reshape(num_samples, -1)
num_features = images_flat.shape[-1]
# # %%
# # save the PCA
# import pickle
# with open('pca_tiny.pkl', 'wb') as file:
#     pickle.dump(pca, file)

# %% 
# Subsample
sub_samples = 20000
indexes = np.random.choice(images_flat.shape[0], sub_samples, replace=False)
images_flat = images_flat[indexes,:]
labels_np = labels_np[indexes]

# %% 
# Project in PCA space
c=pca.transform(images_flat)


# %%
intra_clusters,inter_clusters, ratios, label_dict_transformed, images = {}, {}, {}, {}, {} 
for ratio in [1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.09,0.08,0.07,0.06,0.05,0.01,0.001,0.0001]:
    print("This is ratio",ratio,num_samples,int(ratio * num_samples))

    # do PCA
    i = int(ratio * num_samples)

    for k in list(label_dict.keys()):
        label_dict_transformed[k] = pca.transform(np.reshape(label_dict[k],(label_dict[k].shape[0],-1,)))
        components = np.zeros_like(label_dict_transformed[k])
        components[:, -i:] = label_dict_transformed[k][:, -i:]
        label_dict_transformed[k]=reconstruct_image(pca, components, pca.mean_, [25]+list(original_shape[1:]))

        if k == 0:
            images[ratio] = label_dict_transformed[k][0]


    # intra cluster
    intra_cluster=[]
    for k in list(label_dict_transformed.keys()):
        intra_cluster.append(np.mean(euclidean_distances_jax(label_dict_transformed[k],label_dict_transformed[k])))

    for i, v in enumerate(intra_cluster):
        intra_cluster[i]=v.tolist()
    intra_cluster = np.mean(intra_cluster)

    # inter cluster
    inter_cluster = []
    index = 0
    for k1_index, k1 in enumerate(list(label_dict_transformed.keys())):
        for k2 in list(label_dict_transformed.keys())[(k1_index+1):]:
            if k1 != k2:
                inter_cluster.append(np.mean(euclidean_distances_jax(label_dict_transformed[k1],label_dict_transformed[k2])))

    for i, v in enumerate(inter_cluster):
        inter_cluster[i]=v.tolist()
    inter_cluster = np.mean(inter_cluster)

    # storage
    inter_clusters[ratio]=inter_cluster
    intra_clusters[ratio]=intra_cluster
    ratios[ratio]=round(intra_cluster/inter_cluster,3)

# %% - around 4m 22s per round
alignment_scores = {}
for ratio in [1]: #,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.09,0.08,0.07,0.06,0.05,0.01,0.001,0.0001]:
    print("This is ratio",ratio,num_components,int(ratio * num_components))
    # do PCA
    i = int(ratio * num_components)
    components = np.zeros_like(c)
    components[:, -i:] = c[:, -i:]

    reconstructions=reconstruct_image(pca, components, pca.mean_, [c.shape[0]]+list(original_shape[1:]))
    if len(labels_np.shape)==1:
        labels_np = np.expand_dims(np.array(labels_np),-1)
    reconstructions = np.reshape(reconstructions,[reconstructions.shape[0],-1])
    alignment_scores[ratio] = alignment_sweep(reconstructions,labels_np,major=None)

# %%
alignment_scores_512={}
for k in list(alignment_scores.keys()):
    alignment_scores_512[k]=alignment_scores[k][512]


# %%
granularity = 500
final_component = int(12880*0.94)
nb_plots = len(list(range(1,final_component,granularity)))
nb_plots_row = 6
nb_plots_columns = int(np.ceil(nb_plots/nb_plots_row))

# Function to convert and normalize image from [channels, width, height] to [width, height, channels]
def normalize_and_convert_image(image):

    return np.transpose(image, (1, 2, 0))

# Number of images
num_images = len(images.values())

# Number of rows and columns
num_columns = 5
num_rows = (num_images + num_columns - 1) // num_columns  # This ensures you have enough rows

# Desired size of each image in cm
image_size_cm = 4

# DPI (dots per inch)
dpi = 96  # Adjust based on your display settings

# Convert cm to pixels
image_size_px = image_size_cm * dpi / 2.54

# Create a subplot figure
fig = make_subplots(rows=num_rows, cols=num_columns, subplot_titles=[f'{i}% of PCs' for i in list(images.keys())],
                    horizontal_spacing=0.02, vertical_spacing=0.02)

# Add images to the subplots
for i, image in enumerate(images.values()):
    row = i // num_columns + 1
    col = i % num_columns + 1
    fig.add_trace(
        go.Image(z=normalize_and_convert_image(image*255)),
        row=row, col=col
    )

# Update layout for better spacing
fig.update_layout(height=image_size_px * num_rows, width=image_size_px * num_columns, title_text="Grid of RGB Images")
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

# Show the plot
fig.show()
fig.write_image(f"/local/home/abizeul/reconstruction/notebooks/pc_filtering_{dataset}_{dim}.png")

# %%
# Example data
x = np.arange(len(ratios.values()))
y = list(ratios.values())

# Create a line trace
trace = go.Scatter(
    x=x, 
    y=y, 
    mode='lines+markers',  # Line plot with markers
    line=dict(color='royalblue', width=2),  # Customize line color and width
    marker=dict(color='darkblue', size=6)  # Customize marker color and size
)

# Layout customization
layout = go.Layout(
    title='Does PC filtering impact intra vs. inter cluster distance ?',  # Title of the plot
    title_x=0.5,  # Center the title
    xaxis=dict(
        title='Percentage of PCs kept',  # X-axis title
        tickvals=np.arange(len(ratios.values())),  # Custom tick positions
        ticktext=[int(100*x) for x in list(ratios.keys())],  # Custom tick labels
        showgrid=True,  # Show grid lines
        zeroline=False  # Hide the zero line
    ),
    yaxis=dict(
        title='Ratio intra vs. inter cluster distance in pixel space',  # Y-axis title
        showgrid=True,  # Show grid lines
        zeroline=False  # Hide the zero line
    ),
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Background color of the plot area
    paper_bgcolor='rgba(255, 255, 255, 0.8)',  # Background color of the paper area
)

# Create the figure
fig = go.Figure(data=[trace], layout=layout)

# Show the plot
pio.show(fig)
fig.write_image("pc_filtering_intra_vs_inter.png")

# %%
# Example data
x = np.arange(len(alignment_scores_512.values()))
y = list(alignment_scores_512.values())

# Create a line trace
trace = go.Scatter(
    x=x, 
    y=y, 
    mode='lines+markers',  # Line plot with markers
    line=dict(color='royalblue', width=2),  # Customize line color and width
    marker=dict(color='darkblue', size=6)  # Customize marker color and size
)

# Layout customization
layout = go.Layout(
    title='Does PC filtering impact the alignment score ?',  # Title of the plot
    title_x=0.5,  # Center the title
    xaxis=dict(
        title='Percentage of PCs kept',  # X-axis title
        tickvals=np.arange(len(alignment_scores_512.values())),  # Custom tick positions
        ticktext=[int(100*x) for x in list(alignment_scores_512.keys())],  # Custom tick labels
        showgrid=True,  # Show grid lines
        zeroline=False  # Hide the zero line
    ),
    yaxis=dict(
        title='Alignment',  # Y-axis title
        showgrid=True,  # Show grid lines
        zeroline=False  # Hide the zero line
    ),
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Background color of the plot area
    paper_bgcolor='rgba(255, 255, 255, 0.8)',  # Background color of the paper area
)

# Create the figure
fig = go.Figure(data=[trace], layout=layout)

# Show the plot
pio.show(fig)
fig.write_image(f"/local/home/abizeul/reconstruction/notebooks/pc_filtering_{dataset}_{dim}.png")



# %%
