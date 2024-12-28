#%%
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import medmnist
import torchvision
#%%
derma_dataset=medmnist.DermaMNIST(root="/cluster/project/sachan/callen/data_alice/medmnist",split="train",size=64,transform=torchvision.transforms.ToTensor())
blood_dataset=medmnist.BloodMNIST(root="/cluster/project/sachan/callen/data_alice/medmnist",split="train",size=64,transform=torchvision.transforms.ToTensor())
path_dataset=medmnist.PathMNIST(root="/cluster/project/sachan/callen/data_alice/medmnist",split="train",size=64,transform=torchvision.transforms.ToTensor())

# %%
# Get the first image from each dataset
derma_img = derma_dataset[0][0].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
path_img = path_dataset[1][0].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
blood_img = blood_dataset[0][0].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)

# Create the figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
plt.subplots_adjust(wspace=0.05)  # Adjust this value as needed to bring images closer

# Display the images
axes[0].imshow(derma_img)
axes[0].axis('off')  # Hide axes

axes[1].imshow(path_img)
axes[1].axis('off')  # Hide axes

axes[2].imshow(blood_img)
axes[2].axis('off')  # Hide axes

# Save the figure with a transparent background
plt.savefig('mnist_images.png', transparent=True, dpi=200,bbox_inches='tight')

# Close the plot to free memory
plt.close()
# %%
