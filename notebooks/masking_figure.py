#%% 
import os 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch
from plotly.subplots import make_subplots

#%% 
image1_path = "/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/n01641577/images/n01641577_1.JPEG"
image2_path = "/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/n01855672/images/n01855672_7.JPEG"
image1 = np.transpose(np.array(Image.open(image1_path).convert("RGB")),[2,0,1])/255
image2 = np.transpose(np.array(Image.open(image2_path).convert("RGB")),[2,0,1])/255
pc_matrix = np.load("/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/pc_matrix.npy")
eigenvalues = np.load("/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/eigenvalues_ratio.npy")
eigenvalues_raw = np.load("/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/eigenvalues.npy")
mean = np.load("/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/mean_reshaped.npy") 
std = np.load("/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/std_reshaped.npy")

#%%
def create_mask(image_res,kernel_size,proba,show=False):
    assert kernel_size<=image_res
    ratio = int(np.ceil(image_res/kernel_size))
    nb_events = int(ratio * ratio)
    random_events = round(proba*nb_events)*[0] + round((1-proba)*nb_events)*[1]
    random.shuffle(random_events)
    mask = np.reshape(random_events,[int(ratio),int(ratio)])
    mask = np.kron(mask, np.ones((kernel_size,kernel_size)))
    return mask

def select_pixels(image_res,kernel_size,proba,show=False):
    assert kernel_size<=image_res
    ratio = int(np.ceil(image_res/kernel_size))
    nb_events = int(ratio * ratio)
    random_events = round(proba*nb_events)*[0] + round((1-proba)*nb_events)*[1]
    random.shuffle(random_events)
    mask = np.reshape(random_events,[int(ratio),int(ratio)])
    mask = np.kron(mask, np.ones((kernel_size,kernel_size)))
    mask = (mask-1)*(-1)
    return mask

#%%
mask_75_1 = torch.tensor(create_mask(64,8,0.75)).unsqueeze(0).repeat(3,1,1)
mask_75_2 = torch.tensor(create_mask(64,8,0.75)).unsqueeze(0).repeat(3,1,1)

mask_90_1 = torch.tensor(create_mask(64,8,0.9)).unsqueeze(0).repeat(3,1,1)
mask_90_2 = torch.tensor(create_mask(64,8,0.9)).unsqueeze(0).repeat(3,1,1)

mask_50_1 = torch.tensor(create_mask(64,8,0.5)).unsqueeze(0).repeat(3,1,1)
mask_95_2 = torch.tensor(create_mask(64,8,0.95)).unsqueeze(0).repeat(3,1,1)

#%%
img_75_1 = mask_75_1*image1
img_75_2 = mask_75_2*image2

img_90_1 = mask_90_1*image1
img_90_2 = mask_90_2*image2

img_50_1 = mask_50_1*image1
img_95_2 = mask_95_2*image2

#%% 
# find_threshold = lambda myeigenvalues ,ratio: np.argmin(np.abs(np.cumsum(myeigenvalues) - ratio))
# index = torch.randperm(eigenvalues.shape[0]).numpy()
# threshold = find_threshold(eigenvalues[index],0.85)
# pc_mask = index[:threshold]
# pc_matrix_ocl_t = pc_matrix.T[:,pc_mask] 
# indexes = np.arange(eigenvalues.shape[0])
# pc_mask = indexes[~np.isin(indexes,pc_mask[pc_mask!=-1])]
# pc_matrix_ocl_b = pc_matrix.T[:,pc_mask] 

pc_matrix_ocl_t = pc_matrix.T[:,-int(0.85*pc_matrix.shape[1]):] 
pc_matrix_ocl_b = pc_matrix.T[:,:int(0.15*pc_matrix.shape[1])] 

img1_pc_ocl1 = (np.reshape(((image1-mean)/std).flatten() @ pc_matrix_ocl_b @ pc_matrix_ocl_b.T,[3,64,64])*std)+mean
img1_pc_ocl2 = (np.reshape(((image1-mean)/std).flatten() @ pc_matrix_ocl_t @ pc_matrix_ocl_t.T,[3,64,64])*std)+mean

#%% 
find_threshold = lambda myeigenvalues ,ratio: np.argmin(np.abs(np.cumsum(myeigenvalues) - ratio))
index = torch.randperm(eigenvalues.shape[0]).numpy()
threshold = find_threshold(eigenvalues[index],0.85)
pc_mask = index[:threshold]
pc_matrix_ocl_t = pc_matrix.T[:,pc_mask] 
indexes = np.arange(eigenvalues.shape[0])
pc_mask = indexes[~np.isin(indexes,pc_mask[pc_mask!=-1])]
pc_matrix_ocl_b = pc_matrix.T[:,pc_mask] 

img2_pc_ocl1 = (np.reshape(((image2-mean)/std).flatten() @ pc_matrix_ocl_b @ pc_matrix_ocl_b.T,[3,64,64])*std)+mean
img2_pc_ocl2 = (np.reshape(((image2-mean)/std).flatten() @ pc_matrix_ocl_t @ pc_matrix_ocl_t.T,[3,64,64])*std)+mean

#%% 
find_threshold = lambda myeigenvalues ,ratio: np.argmin(np.abs(np.cumsum(myeigenvalues) - ratio))
index = torch.randperm(eigenvalues.shape[0]).numpy()
threshold = find_threshold(eigenvalues[index],np.random.randint(10,90,1)[0]/100)
pc_mask = index[:threshold]
pc_matrix_ocl_t = pc_matrix.T[:,pc_mask] 
indexes = np.arange(eigenvalues.shape[0])
pc_mask = indexes[~np.isin(indexes,pc_mask[pc_mask!=-1])]
pc_matrix_ocl_b = pc_matrix.T[:,pc_mask] 

img1_pc_rd1 = (np.reshape(((image1-mean)/std).flatten() @ pc_matrix_ocl_b @ pc_matrix_ocl_b.T,[3,64,64])*std)+mean
img1_pc_rd2 = (np.reshape(((image1-mean)/std).flatten() @ pc_matrix_ocl_t @ pc_matrix_ocl_t.T,[3,64,64])*std)+mean

#%% 
pc_matrix_ocl_b = pc_matrix.T[:,-int(0.80*pc_matrix.shape[1]):] 
pc_matrix_ocl_t = pc_matrix.T[:,:int(0.20*pc_matrix.shape[1])] 

img2_pc_rd1 = (np.reshape(((image2-mean)/std).flatten() @ pc_matrix_ocl_b @ pc_matrix_ocl_b.T,[3,64,64])*std)+mean
img2_pc_rd2 = (np.reshape(((image2-mean)/std).flatten() @ pc_matrix_ocl_t @ pc_matrix_ocl_t.T,[3,64,64])*std)+mean

#%%
image1 = np.transpose(image1,[1,2,0])
image2 = np.transpose(image2,[1,2,0])
image_75_1 = np.transpose(img_75_1,[1,2,0])
image_75_2 = np.transpose(img_75_2,[1,2,0])
image_90_1 = np.transpose(img_90_1,[1,2,0])
image_90_2 = np.transpose(img_90_2,[1,2,0])
image_50_1 = np.transpose(img_50_1,[1,2,0])
image_95_2 = np.transpose(img_95_2,[1,2,0])
img1_pc_ocl1 = np.transpose(img1_pc_ocl1,[1,2,0])
img1_pc_ocl2 = np.transpose(img1_pc_ocl2,[1,2,0])
img2_pc_ocl1 = np.transpose(img2_pc_ocl1,[1,2,0])
img2_pc_ocl2 = np.transpose(img2_pc_ocl2,[1,2,0])
img1_pc_rd1 = np.transpose(img1_pc_rd1,[1,2,0])
img1_pc_rd2 = np.transpose(img1_pc_rd2,[1,2,0])
img2_pc_rd1 = np.transpose(img2_pc_rd1,[1,2,0])
img2_pc_rd2 = np.transpose(img2_pc_rd2,[1,2,0])

#%%
# Create a figure with transparent background
fig, axes = plt.subplots(2, 8, figsize=(24, 6), constrained_layout=True)
fig.patch.set_alpha(0)  # Make the background transparent

# Row 1
axes[0, 0].imshow(image1, cmap='gray')
axes[0, 0].axis('off')
# axes[0, 0].set_title("Image 1 (75%)")

# Row 1
axes[1, 0].imshow(image2, cmap='gray')
axes[1, 0].axis('off')
# axes[0, 0].set_title("Image 1 (75%)")

# Row 1
axes[0, 1].imshow(image_75_1, cmap='gray')
axes[0, 1].axis('off')
# axes[0, 0].set_title("Image 1 (75%)")

axes[1, 1].imshow(image_75_2, cmap='gray')
axes[1, 1].axis('off')
# axes[0, 1].set_title("Image 2 (75%)")

axes[0, 2].imshow(image_90_1, cmap='gray')
axes[0, 2].axis('off')
# axes[0, 2].set_title("Image 1 (90%)")

# Row 2
axes[1, 2].imshow(image_90_2, cmap='gray')
axes[1, 2].axis('off')
# axes[1, 0].set_title("Image 2 (90%)")

axes[0, 3].imshow(image_50_1, cmap='gray')
axes[0, 3].axis('off')
# axes[1, 1].set_title("Image 1 (50%)")

axes[1, 3].imshow(image_95_2, cmap='gray')
axes[1, 3].axis('off')

####
axes[0, 4].imshow(img1_pc_ocl1, cmap='gray')
axes[0, 4].axis('off')

axes[1, 4].imshow(img2_pc_ocl1, cmap='gray')
axes[1, 4].axis('off')

axes[0, 5].imshow(img1_pc_ocl2, cmap='gray')
axes[0, 5].axis('off')

axes[1, 5].imshow(img2_pc_ocl2, cmap='gray')
axes[1, 5].axis('off')

####
axes[0, 6].imshow(img1_pc_rd1, cmap='gray')
axes[0, 6].axis('off')

axes[1, 7].imshow(img2_pc_rd1, cmap='gray')
axes[1, 7].axis('off')

axes[0, 7].imshow(img1_pc_rd2, cmap='gray')
axes[0, 7].axis('off')

axes[1, 6].imshow(img2_pc_rd2, cmap='gray')
axes[1, 6].axis('off')

fig.savefig('masked_images.png', transparent=True, dpi=200,bbox_inches='tight')

plt.show()
# %%
import plotly.graph_objects as go
import numpy as np

# Load eigenvalues
eigenvalues = np.load("/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/eigenvalues_ratio.npy")
eigenvalues_raw = np.load("/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/eigenvalues.npy")

# Ensure eigenvalues_raw doesn't contain zero or negative values for log calculation
eigenvalues_raw[eigenvalues_raw <= 0] = 1e-10  # Replace non-positive values to avoid log errors

# Create the figure
fig = go.Figure()
plotly_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

# 50,500,1000,1500
partitions = [1,1,int(0.05*eigenvalues.shape[0]), int(0.10*eigenvalues.shape[0]), eigenvalues.shape[0]]  # Adjust these x-values as needed
colors = ['rgba(99, 110, 250, 0.2)', 'rgba(99, 110, 250, 0.2)', 'rgba(99, 110, 250, 0.5)', 'rgba(99, 110, 250, 0.8)']


# Add the first curve (y1 axis with log scale)
fig.add_trace(go.Scatter(
    x=np.arange(1, eigenvalues_raw.shape[0] + 1), 
    y=eigenvalues_raw,  # Logarithmic transformation
    mode='lines',
    name='Eigenvalue',  # Add legend for the first curve
    line=dict(color="gray"),  # Set color for the first curve
    showlegend=True,  # Show legend for this curve
    yaxis='y2',  # Specify that this curve should use the second y-axis
))

for i in range(4):
    start = partitions[i]
    end = partitions[i + 1]
    print(colors[i],start,end,eigenvalues.shape)

    fig.add_trace(go.Scatter(
        x=np.arange(start, end + 1),
        y=eigenvalues_raw[start - 1:end], 
        mode='lines',
        name='Eigenvalue',  # Add legend for the first curve
        line=dict(color="gray"),  # Set color for the first curve
        showlegend=False,  # Show legend for this curve
        yaxis='y2',  # Line color
        fill='tozeroy',  # Fill the area to the x-axis
        fillcolor=colors[i],  # Apply the color with varying opacities
    ))

# Add the second curve (y2 axis with cumulative values)
fig.add_trace(go.Scatter(
    x=np.arange(1, eigenvalues.shape[0] + 1), 
    y=np.cumsum(eigenvalues), 
    mode='lines',
    name="Cum. Sum % Var. Expl.",  # Add legend for the second curve
    line=dict(color="gray",dash="dash"),  # Set color for the second curve
    showlegend=True  # Show legend for this curve
))

# Update layout to include a second y-axis on the right side, adjust the font size, and change the figure size
fig.update_layout(
    xaxis_title="Eigenvalues Index",
    yaxis_title="Cum. Sum of %  of Var. Expl.",
    yaxis=dict(
        title="Cum. Sum of %  of Var. Expl.",
        showgrid=False,  # Disable grid lines for the second axis
        showline=True
    ),
    yaxis2=dict(
        title="Eigenvalue",  # Log scale for the first curve
        showline=True,
        overlaying='y',  # Overlay on the same plot
        side='right',  # Set the second y-axis to the right side
        zeroline=False,  # Hide the zero line on the log scale
        type="log",  # Set the first y-axis to logarithmic scale
    ),
    width=600,  # Adjust the width of the figure (in pixels)
    height=400,  # Adjust the height of the figure (in pixels)
    font=dict(
        size=18,  # Increase the font size for axis titles, ticks, and legend
    ),
    margin=dict(l=20, r=20, t=40, b=20),  # Tighten the margins around the figure
    legend=dict(
        x=0.375,  # Adjust the position of the legend (left)
        y=0.785,  # Adjust the position of the legend (top)
        bgcolor="rgba(255,255,255,0.8)",  # Set the background of the legend
        borderwidth=0  # Set the border width of the legend
    ),
    # plot_bgcolor='rgba(0, 0, 0, 0)',  # Set the plot area background to transparent
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Set the entire figure background to transparent
)

fig.update_xaxes(gridcolor="gray")
fig.update_yaxes(gridcolor="gray")

# Save the figure with tight margins around the plot content
fig.write_image("./eigenvalues_analysis_tight.pdf", scale=2, width=600, height=400)

# Show the figure
fig.show()

# %%
import plotly.graph_objects as go
import numpy as np

# Load eigenvalues
eigenvalues = np.load("/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/eigenvalues_ratio.npy")
eigenvalues_raw = np.load("/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/eigenvalues.npy")

# Ensure eigenvalues_raw doesn't contain zero or negative values for log calculation
eigenvalues_raw[eigenvalues_raw <= 0] = 1e-10  # Replace non-positive values to avoid log errors

# Create the figure
fig = go.Figure()
plotly_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

# 50,500,1000,1500
partitions = [1,1,int(0.05*eigenvalues.shape[0]), int(0.10*eigenvalues.shape[0]), eigenvalues.shape[0]]  # Adjust these x-values as needed
colors = ['rgba(99, 110, 250, 0.2)', 'rgba(99, 110, 250, 0.2)', 'rgba(99, 110, 250, 0.5)', 'rgba(99, 110, 250, 0.8)']


# Add the first curve (y1 axis with log scale)
fig.add_trace(go.Scatter(
    x=np.arange(1, eigenvalues_raw.shape[0] + 1), 
    y=eigenvalues_raw,  # Logarithmic transformation
    mode='lines',
    name='Eigenvalue',  # Add legend for the first curve
    line=dict(color="gray"),  # Set color for the first curve
    showlegend=True,  # Show legend for this curve
    yaxis='y2',  # Specify that this curve should use the second y-axis
))

# Add the second curve (y2 axis with cumulative values)
fig.add_trace(go.Scatter(
    x=np.arange(1, eigenvalues.shape[0] + 1), 
    y=np.cumsum(eigenvalues), 
    mode='lines',
    name="Cum. Sum % Var. Expl.",  # Add legend for the second curve
    line=dict(color="gray",dash="dash"),  # Set color for the second curve
    showlegend=True  # Show legend for this curve
))

# Update layout to include a second y-axis on the right side, adjust the font size, and change the figure size
fig.update_layout(
    xaxis_title="Eigenvalues Index",
    yaxis_title="Cum. Sum of %  of Var. Expl.",
    yaxis=dict(
        title="Cum. Sum of %  of Var. Expl.",
        showgrid=False,  # Disable grid lines for the second axis
        showline=True
    ),
    yaxis2=dict(
        title="Eigenvalue",  # Log scale for the first curve
        showline=True,
        overlaying='y',  # Overlay on the same plot
        side='right',  # Set the second y-axis to the right side
        zeroline=False,  # Hide the zero line on the log scale
        type="log",  # Set the first y-axis to logarithmic scale
    ),
    width=600,  # Adjust the width of the figure (in pixels)
    height=400,  # Adjust the height of the figure (in pixels)
    font=dict(
        size=18,  # Increase the font size for axis titles, ticks, and legend
    ),
    margin=dict(l=20, r=20, t=40, b=20),  # Tighten the margins around the figure
    legend=dict(
        x=0.375,  # Adjust the position of the legend (left)
        y=0.78,  # Adjust the position of the legend (top)
        bgcolor="rgba(255,255,255,0.8)",  # Set the background of the legend
        borderwidth=0  # Set the border width of the legend
    ),
    # plot_bgcolor='rgba(0, 0, 0, 0)',  # Set the plot area background to transparent
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Set the entire figure background to transparent
)

fig.update_xaxes(gridcolor="gray")
fig.update_yaxes(gridcolor="gray")

# Save the figure with tight margins around the plot content
fig.write_image("./eigenvalues_analysis_tight_blank.pdf", scale=2, width=600, height=400)

# Show the figure
fig.show()

#%%
find_threshold = lambda myeigenvalues ,ratio: np.argmin(np.abs(np.cumsum(myeigenvalues) - ratio))
index = torch.randperm(eigenvalues.shape[0]).numpy()
ratio=0.25
threshold = find_threshold(eigenvalues[index],ratio)

# Create the figure
fig = go.Figure()
plotly_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

# Add the second curve (y2 axis with cumulative values)
fig.add_trace(go.Scatter(
    x=np.arange(1, eigenvalues.shape[0] + 1), 
    y=np.cumsum(eigenvalues[index]), 
    mode='lines',
    name="Cum. Sum % Var. Expl.",  # Add legend for the second curve
    line=dict(color="gray",dash="dash"),  # Set color for the second curve
    showlegend=True,  # Show legend for this curve
))

fig.add_trace(go.Scatter(
    x=np.arange(1, eigenvalues.shape[0] + 1), 
    y=eigenvalues_raw[index], 
    mode='lines',
    name="Eigenvalue",  # Add legend for the second curve
    line=dict(color="gray"),  # Set color for the second curve
    showlegend=True,  # Show legend for this curve
    yaxis='y2',  # Specify that this curve should use the second y-axis
))


# Update layout to include a second y-axis on the right side, adjust the font size, and change the figure size
fig.update_layout(
    xaxis_title="Shuffled Eigenvalues Index",
    yaxis_title="Cum. Sum of %  of Var. Expl.",
    width=600,  # Adjust the width of the figure (in pixels)
    height=400,  # Adjust the height of the figure (in pixels)
    font=dict(
        size=18,  # Increase the font size for axis titles, ticks, and legend
    ),
    margin=dict(l=20, r=20, t=40, b=20),  # Tighten the margins around the figure
    legend=dict(
        x=0.03, #0.05,  # Adjust the position of the legend (left)
        y=0.57, #0.92,  # Adjust the position of the legend (top)
        bgcolor="rgba(255,255,255,0.5)",  # Set the background of the legend
        borderwidth=0  # Set the border width of the legend
    ),
    yaxis=dict(showgrid=True, gridcolor="gray"),  # Enable grid for the first y-axis
    yaxis2=dict(title="Eigenvalue",showgrid=False, overlaying='y', side='right'),
    # plot_bgcolor='rgba(0, 0, 0, 0)',  # Set the plot area background to transparent
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Set the entire figure background to transparent
)

fig.update_xaxes(gridcolor="gray",tickvals=np.arange(1, eigenvalues.shape[0] + 1)[::1000],  ticktext=index[::1000])


# Save the figure with tight margins around the plot content
fig.write_image("./eigenvalues_analysis_shuffling.png", scale=2, width=600, height=400)

# Show the figure
fig.show()

#%%
# Create the figure
fig = go.Figure()
plotly_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

# Add the second curve (y2 axis with cumulative values)
fig.add_trace(go.Scatter(
    x=np.arange(1, eigenvalues.shape[0] + 1), 
    y=np.cumsum(eigenvalues[index]), 
    mode='lines',
    name="Cum. Sum % Var. Expl.",  # Add legend for the second curve
    line=dict(color="gray",dash="dash"),  # Set color for the second curve
    showlegend=True,  # Show legend for this curve
))

fig.add_trace(go.Scatter(
    x=np.arange(1, eigenvalues.shape[0] + 1), 
    y=eigenvalues_raw[index], 
    mode='lines',
    name="Eigenvalue",  # Add legend for the second curve
    line=dict(color="black"),  # Set color for the second curve
    showlegend=True,  # Show legend for this curve
    yaxis='y2',  # Specify that this curve should use the second y-axis
))

# Add a vertical line at the specified x-value
fig.add_shape(
    type="line",
    x0=0,
    y0=ratio,
    x1=eigenvalues.shape[0],
    y1=ratio,  # The height of the vertical line should match the highest point in the data
    line=dict(color=plotly_colors[1], width=2, dash="dash"),  # Customize line style
)

colors = ['rgba(99, 110, 250, 0.4)', 'rgba(99, 110, 250, 0.8)', 'rgba(99, 110, 250, 0.5)', 'rgba(99, 110, 250, 0.8)']
partitions=[1,threshold,eigenvalues.shape[0]]
for i in range(2):
    start = partitions[i]
    end = partitions[i + 1]
    fig.add_trace(go.Scatter(
        x=np.arange(start, end + 1),
        y=np.cumsum(eigenvalues[index])[start - 1:end], 
        mode='lines',
        line=dict(color="gray",dash="dash"),  # Set color for the first curve
        showlegend=False,  # Show legend for this curve
        fill='tozeroy',  # Fill the area to the x-axis
        fillcolor=colors[i],  # Apply the color with varying opacities
    ))


# Update layout to include a second y-axis on the right side, adjust the font size, and change the figure size
fig.update_layout(
    xaxis_title="Shuffled Eigenvalues Index",
    yaxis_title="Cum. Sum of %  of Var. Expl.",
    width=600,  # Adjust the width of the figure (in pixels)
    height=400,  # Adjust the height of the figure (in pixels)
    font=dict(
        size=18,  # Increase the font size for axis titles, ticks, and legend
    ),
    margin=dict(l=20, r=20, t=40, b=20),  # Tighten the margins around the figure
    legend=dict(
        x=0.03,  # Adjust the position of the legend (left)
        y=0.87,  # Adjust the position of the legend (top)
        bgcolor="rgba(255,255,255,0.9)",  # Set the background of the legend
        borderwidth=0  # Set the border width of the legend
    ),
    yaxis=dict(showgrid=True, gridcolor="gray"),  # Enable grid for the first y-axis
    yaxis2=dict(title="Eigenvalue",showgrid=False, overlaying='y', side='right'),
    # plot_bgcolor='rgba(0, 0, 0, 0)',  # Set the plot area background to transparent
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Set the entire figure background to transparent
)

fig.update_xaxes(gridcolor="gray",showticklabels=False,title_text="Shuffled Eigenvalues Index")


# Save the figure with tight margins around the plot content
fig.write_image("./eigenvalues_analysis_masking.pdf", scale=2, width=600, height=400)

# Show the figure
fig.show()

#%%
find_threshold = lambda myeigenvalues ,ratio: np.argmin(np.abs(np.cumsum(myeigenvalues) - ratio))
index = torch.randperm(eigenvalues.shape[0]).numpy()
ratio=0.25
threshold = find_threshold(eigenvalues[index],ratio)

# Create the figure
fig = go.Figure()
plotly_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

# Add the second curve (y2 axis with cumulative values)
fig.add_trace(go.Scatter(
    x=np.arange(1, eigenvalues.shape[0] + 1), 
    y=np.cumsum(eigenvalues[index]), 
    mode='lines',
    name="Cum. Sum % Var. Expl.",  # Add legend for the second curve
    line=dict(color="gray"),  # Set color for the second curve
    showlegend=True,  # Show legend for this curve
))

# Add a vertical line at the specified x-value
fig.add_shape(
    type="line",
    x0=0,
    y0=ratio,
    x1=eigenvalues.shape[0],
    y1=ratio,  # The height of the vertical line should match the highest point in the data
    line=dict(color=plotly_colors[1], width=2, dash="dash"),  # Customize line style
)

fig.add_shape(
    **{
        'type': 'rect',
        'xref': 'x',
        'yref': 'y',
        'x0': 0,
        'y0': 0,
        'x1': threshold,  # Left-hand side rectangle ends at the threshold
        'y1': 1.05,  # Color up to the maximum y value
        'fillcolor': '#636EFA',  # Fill color for the left-hand side
        'opacity': 0.2,
        'layer': 'below',  # Place the fill below the curves
        'line_width': 0,
    }
)

fig.add_shape(
    **{
        'type': 'rect',
        'xref': 'x',
        'yref': 'y',
        'x0': threshold,  # Right-hand side rectangle starts at the threshold
        'y0': 0,
        'x1': eigenvalues.shape[0],
        'y1': 1.05,  # Color up to the maximum y value
        'fillcolor': '#636EFA',  # Different fill color for the right-hand side
        'opacity': 0.6,
        'layer': 'below',  # Place the fill below the curves
        'line_width': 0,
    }
)

# Update layout to include a second y-axis on the right side, adjust the font size, and change the figure size
fig.update_layout(
    xaxis_title="Eigenvalues Index",
    yaxis_title="Cum. Sum of %  of Var. Explained",
    width=600,  # Adjust the width of the figure (in pixels)
    height=400,  # Adjust the height of the figure (in pixels)
    font=dict(
        size=16,  # Increase the font size for axis titles, ticks, and legend
    ),
    margin=dict(l=20, r=20, t=40, b=20),  # Tighten the margins around the figure
    legend=dict(
        x=0.45,  # Adjust the position of the legend (left)
        y=0.1,  # Adjust the position of the legend (top)
        bgcolor="rgba(255,255,255,0.5)",  # Set the background of the legend
        borderwidth=0  # Set the border width of the legend
    ),
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Set the plot area background to transparent
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Set the entire figure background to transparent
)

fig.update_yaxes(gridcolor="gray")
fig.update_xaxes(gridcolor="gray")

# Save the figure with tight margins around the plot content
fig.write_image("./eigenvalues_analysis_tight2.png", scale=2, width=600, height=400)

# Show the figure
fig.show()
# %%
