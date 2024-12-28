import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os 
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

# Load mean and std from the specified numpy files
mean = torch.tensor(np.load('/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/mean_reshaped.npy'))
std = torch.tensor(np.load('/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/std_reshaped.npy'))

# Define the transformations
transform = transforms.Compose([
    # transforms.RandomResizedCrop((64, 64), scale=(0.2, 1.0), interpolation=3),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Create the CIFAR-10 dataset with the defined transformations
# cifar10_dataset = torchvision.datasets.CIFAR10(
#     root='/cluster/project/sachan/callen/data_alice',
#     train=True,
#     download=True,
#     transform=transform
# )
# Create the TinyImageNet dataset with the defined transformations
cifar10_dataset = ImageFolder(
    root='/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train',
    transform=transform
)

pca = np.load("/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/train/pc_matrix.npy")

print(pca.shape)

# Create directories to store the images
os.makedirs("images_before_pca", exist_ok=True)
os.makedirs("images_after_pca", exist_ok=True)

# Define the DataLoader
dataloader = torch.utils.data.DataLoader(
    cifar10_dataset,
    batch_size=1,  # You can adjust the batch size as needed
    shuffle=False,
    num_workers=4   # Adjust the number of workers as needed for your system
)

losses = []
total_mse = 0.0
num_images = 0
# Iterate through the DataLoader and save the images
for i, (batch, label) in enumerate(dataloader):
    # Reshape the batch to a 2D tensor
    batch = torch.reshape(batch, (batch.shape[0], -1))
    
    # Perform PCA transformation
    batch_reconstruction = batch @ torch.tensor(pca) @ torch.tensor(pca.T)

    # Compute the Mean Squared Error
    mse = torch.mean((batch - batch_reconstruction) ** 2).item()
    total_mse += mse
    num_images += 1

    
    # Reshape the images back to their original shape (3, 32, 32)
    batch_image = torch.reshape(batch, (1, 3, 64, 64))
    reconstructed_image = torch.reshape(batch_reconstruction, (1, 3, 64, 64))
    
    # Save the images before and after PCA transformation
    save_image(batch_image, f"images_before_pca/image_{i}.png")
    save_image(reconstructed_image, f"images_after_pca/image_{i}.png")
    
    if i >= 100:  # Save only a few images for quick testing
        print("This is the loss",total_mse / num_images)
        break
