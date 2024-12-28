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
# alignment metric
def alignment_sweep(X, Y, ratios, dim=512, save_path="",major="C"): 
    # here we consider that N > D and that X lies in NxD, for D > N change to major "column"
     
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            U = pickle.load(f)
    else:
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        with open(save_path, 'wb') as f:
            pickle.dump(U, f)

    alignments, normalization, fig = {}, {}, {}
    for ratio in ratios:

        # if major == "C": 
        #     # denom = np.square(np.linalg.norm(Y @ Y.T)) 
        #     numer = np.linalg.multi_dot([Y.T, Y, Vh.T]) 
        # else: 
            # denom = np.square(np.linalg.norm(Y.T @ Y)) 
        num_components = U.shape[-1]
        i = int(ratio * num_components)
        # components = np.zeros_like(U)
        # components[:, -i:] = U[:, -i:]
        # numer = np.linalg.multi_dot([Y,Y.T, U])
        numer = np.linalg.multi_dot([Y,Y.T,U[:, -i:]])  
        numer = np.linalg.norm(numer, axis=0)**2 
        # fig[ratio]=numer
        numer = np.cumsum(numer)
        if ratio ==1:
            reference = numer[-1]
        fig[ratio] = numer/numer[-1]
        alignments[ratio] = numer[dim]/reference
        normalization[ratio] = numer[-1]/reference
        
    return alignments, normalization, fig

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
    # pca = PCA()  # You can adjust the number of components
    # pca.fit(images_flat)
    # c=pca.transform(images_flat)

    # %%
    # Eval data
    # trainloader =  torch.utils.data.DataLoader(testset, batch_size=len(trainset), shuffle=False, num_workers=4)

    # Fetch the entire dataset in one go
    # data_iter = iter(trainloader)
    # images, labels = next(data_iter)

    # Step 2: Convert the dataset to NumPy arrays
    # images_np = images.numpy()
    # labels_np = labels.numpy()

    # Reshape the images to (num_samples, height * width * channels)
    # num_samples = images_np.shape[0]
    # original_shape = images_np.shape
    # images_flat = images_np.reshape(num_samples, -1)
    # num_features = images_flat.shape[-1]

    # Standardize
    # mean, std = np.mean(images_flat, axis=0), np.std(images_flat, axis=0)
    # images_flat = ((images_flat - mean)/std)
    # mean, std = np.mean(labels_np, axis=0), np.std(labels_np, axis=0)
    # labels_np = ((labels_np - mean)/std)

    # %% 
    # Subsample
    # images_flat = images_flat[indexes,:]
    # labels_np = labels_np[indexes]

    # %% 
    # Project in PCA space
    # c=pca.transform(images_flat)

    # %% - around 4m 22s per round
    # alignment_scores, normalization_scores = {}, {}
    images = {}
    images[0] = np.reshape(images_flat[0],list(original_shape[1:]))
    ratios = [1,0.85,0.55]
    # for ratio in [1,0.85,0.55]:
    #     print("This is ratio",ratio,num_components,int(ratio * num_components),c.shape)
    #     # do PCA
    #     i = int(ratio * num_components)
    #     components = np.zeros_like(c)
    #     components[:, -i:] = c[:, -i:]

    #     reconstructions=reconstruct_image(pca, components, pca.mean_, [c.shape[0]]+list(original_shape[1:]))
    # if len(labels_np.shape)==1:
        # labels_np = np.expand_dims(np.array(labels_np),-1)
        # images[ratio]=reconstructions[0]
        # reconstructions = np.reshape(reconstructions,[reconstructions.shape[0],-1])

        # # standardize again 
        # mean, std = np.mean(reconstructions, axis=0), np.std(reconstructions, axis=0)
        # reconstructions = ((reconstructions - mean)/std)

        # compute alignment
    alignment_scores, normalization_scores, fig_scores = alignment_sweep(images_flat,labels_np,ratios,save_path="/local/home/abizeul/reconstruction/notebooks/svd_u_tiny_15k.pkl",major=None)

    # %%
    # alignment_scores_dim={}
    # for k in list(alignment_scores.keys()):
    #     alignment_scores_dim[k]=alignment_scores[k][dim]

    # %% compute final normalisation scores
    # for k in list(normalization_scores.keys()):
    #     normalization_scores[k]=normalization_scores[k]/normalization_scores[1]

    # %% save images
    granularity = 500
    final_component = int(12880*0.94)
    nb_plots = len(list(range(1,final_component,granularity)))
    nb_plots_row = 6
    # nb_plots_columns = int(np.ceil(nb_plots/nb_plots_row))

    # Function to convert and normalize image from [channels, width, height] to [width, height, channels]
    def normalize_and_convert_image(image):
        return np.transpose(image, (1, 2, 0))

    # Number of images
    num_images = len(images.values())
    print("This is the number of images",num_images)

    # Number of rows and columns
    num_columns = 5
    num_rows = (num_images + num_columns - 1) // num_columns  # This ensures you have enough rows

    # Desired size of each image in cm
    image_size_cm = 8

    # DPI (dots per inch)
    dpi = 96  # Adjust based on your display settings

    # Convert cm to pixels
    image_size_px = image_size_cm * dpi / 2.54

    # Create a subplot figure
    fig = make_subplots(rows=num_rows, cols=num_columns, subplot_titles=[f'{int(100*i)}% of PCs' for i in list(images.keys())],
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
    # fig.show()
    fig.write_image(f"pc_filtering_images_{dataset}.png")

    # %%
    # Example data
    x = np.arange(len(alignment_scores.values()))
    y = list(alignment_scores.values())

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
            tickvals=np.arange(len(alignment_scores.values())),  # Custom tick positions
            ticktext=[int(100*x) for x in list(alignment_scores.keys())],  # Custom tick labels
            showgrid=True,  # Show grid lines
            zeroline=False  # Hide the zero line
        ),
        yaxis=dict(
            title='Alignment of first 512 dimensions',  # Y-axis title
            showgrid=True,  # Show grid lines
            zeroline=False  # Hide the zero line
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Background color of the plot area
        paper_bgcolor='rgba(255, 255, 255, 0.8)',  # Background color of the paper area
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)
    fig.write_image(f"/local/home/abizeul/reconstruction/notebooks/pc_filtering_{dataset}_{dim}.png")


    # %%
    # Example data
    x = np.arange(len(normalization_scores.values()))
    y = list(normalization_scores.values())

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
            tickvals=np.arange(len(normalization_scores.values())),  # Custom tick positions
            ticktext=[int(100*x) for x in list(normalization_scores.keys())],  # Custom tick labels
            showgrid=True,  # Show grid lines
            zeroline=False  # Hide the zero line
        ),
        yaxis=dict(
            title='Ratio of normalisation scores (amount of Y variance kept in PC projection)',  # Y-axis title
            showgrid=True,  # Show grid lines
            zeroline=False  # Hide the zero line
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Background color of the plot area
        paper_bgcolor='rgba(255, 255, 255, 0.8)',  # Background color of the paper area
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)
    fig.write_image(f"/local/home/abizeul/reconstruction/notebooks/ratio_normalisation_{dataset}_{dim}.png")


    # Show the plot

    #pio.show(fig)
    # fig.write_image(f"/local/home/abizeul/reconstruction/notebooks/pc_filtering_{dataset}_{dim}.png")

    # %%
    # Example data
    x = list(alignment_scores.values())
    y = list(normalization_scores.values())
    labels = [int(100*x) for x in list(alignment_scores.keys())]

    # Create a line trace
    trace = go.Scatter(
        x=x, 
        y=y, 
        mode='markers+text',  # Scatter plot with text
        text=labels,  # Text labels
        textposition='top right',  # Position of the text
        # line=dict(color='royalblue', width=2),  # Customize line color and width
        marker=dict(color='darkblue', size=6)  # Customize marker color and size
    )

    # Layout customization
    layout = go.Layout(
        title='Can we find tradeoff between performance and efficiency ?',  # Title of the plot
        title_x=0.5,  # Center the title
        xaxis=dict(
            title='Alignment of first 512 dimensions',  # X-axis title
            # tickvals=np.arange(len(normalization_scores.values())),  # Custom tick positions
            # ticktext=[int(100*x) for x in list(normalization_scores.keys())],  # Custom tick labels
            showgrid=True,  # Show grid lines
            zeroline=False  # Hide the zero line
        ),
        yaxis=dict(
            title='Ratio of normalisation values (amount of Y variance kept in PC projection)',  # Y-axis title
            showgrid=True,  # Show grid lines
            zeroline=False  # Hide the zero line
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Background color of the plot area
        paper_bgcolor='rgba(255, 255, 255, 0.8)',  # Background color of the paper area
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)
    fig.write_image(f"/local/home/abizeul/reconstruction/notebooks/tradeoff_{dataset}_{dim}.png")


# %%
    # Example data

    # Create a line trace
    traces = []
    for ratio in list(fig_scores.keys()):
        x = np.arange(len(fig_scores[ratio]))
        y = list(fig_scores[ratio])

        traces.append(go.Scatter(
            x=x, 
            y=y, 
            mode='lines',  # Line plot with markers
            line=dict(width=2),   # Customize line color and width
            name=f"{str(int(100*ratio))}% of PCs"
            # marker=dict(color='darkblue', size=6)  # Customize marker color and size
        ))

    # Layout customization
    layout = go.Layout(
        title='Reproducing figure 2 for different projections in PC space',  # Title of the plot
        title_x=0.5,  # Center the title
        xaxis=dict(
            title='Latent dimension',  # X-axis title
            # tickvals=np.arange(len(normalization_scores.values())),  # Custom tick positions
            # ticktext=[int(100*x) for x in list(normalization_scores.keys())],  # Custom tick labels
            showgrid=True,  # Show grid lines
            zeroline=False  # Hide the zero line
        ),
        yaxis=dict(
            title='Alignment score',  # Y-axis title
            showgrid=True,  # Show grid lines
            zeroline=False  # Hide the zero line
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Background color of the plot area
        paper_bgcolor='rgba(255, 255, 255, 0.8)',  # Background color of the paper area
    )

    # Create the figure
    fig = go.Figure(data=traces,layout=layout)
    fig.write_image(f"/local/home/abizeul/reconstruction/notebooks/alignment_{dataset}.png")

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A more complex script example.")
    
    # Required positional argument
    parser.add_argument('--dataset', type=str, default="tinyimagenet", help='The file to process.')
    parser.add_argument('--dim',     type=int, default=512, help='Number of times to process the file.')

    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
