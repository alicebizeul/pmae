from scipy.fftpack import dct, idct
import torch
from torchvision import datasets, transforms
import random
from sklearn.metrics import mean_squared_error
import numpy as np
import os 
from PIL import Image

# Create directories for saving images
os.makedirs('/cluster/home/abizeul/mae/tools/dct/batch', exist_ok=True)
os.makedirs('/cluster/home/abizeul/mae/tools/dct/coefs', exist_ok=True)
os.makedirs('/cluster/home/abizeul/mae/tools/dct/filtered', exist_ok=True)
os.makedirs('/cluster/home/abizeul/mae/tools/dct/batch_recon', exist_ok=True)

# Define the data transformations
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a fixed size
    transforms.CenterCrop(224),     # Crop to the target size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Path to the ImageNet data
imagenet_data_path = '/cluster/project/sachan/callen/data_alice/ILSVRC2012_img/val/ILSVRC2012'

# Create the ImageFolder dataset
imagenet_dataset = datasets.ImageFolder(root=imagenet_data_path, transform=data_transforms)

# Create a DataLoader for the subset
dataloader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=32, shuffle=True, num_workers=4)

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho',type=1).T, norm='ortho',type=2)

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho',type=1).T, norm='ortho',type=2)    


coefs,mse = [],[]
order = np.load("/cluster/home/abizeul/mae/tools/dct/ordered_dct.npy")
for i, (batch, _) in enumerate(dataloader):
    # Convert batch to numpy and de-normalize
    batch = batch.numpy()

    # Save images from the batch
    for j, b in enumerate(batch):
        img = (b * np.array([0.229, 0.224, 0.225])[:, None, None] + np.array([0.485, 0.456, 0.406])[:, None, None])
        img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Convert to [0, 255] range
        # img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        # Image.fromarray(np.transpose(img, (1, 2, 0))).save(f'/cluster/home/abizeul/mae/tools/dct/batch/image_{i}_{j}.png')

        coef = np.stack([dct2(b[0,:,:]),dct2(b[1,:,:]),dct2(b[2,:,:])],0)
        # order = np.argsort(np.mean(coef,0).flatten())[::-1]
        # coefs.append(coef)
        # batch_recon = np.stack([idct2(coef[0]),idct2(coef[1]),idct2(coef[2])],0)

        # Save DCT coefficients as images (normalize to [0, 255])
        # coef_norm = np.clip((coef - coef.min()) / (coef.max() - coef.min()) * 255, 0, 255).astype(np.uint8)
        # print("coef norm",coef_norm.shape,b.shape,batch_recon.shape)
        # np.save(f'/cluster/home/abizeul/mae/tools/dct/coefs/dct_coef_{i}_{j}',coef)
        # img = Image.fromarray(np.transpose(coef_norm, (1, 2, 0)))
        # img.save(f'/cluster/home/abizeul/mae/tools/dct/coefs/image_{i}_{j}.png')

        # Calculate MSE
        # mse.append(mean_squared_error(b.flatten(),batch_recon.flatten()))

        # Save reconstructed images
        # print("shape",batch_recon.shape)
        # batch_recon = (batch_recon * np.array([0.229, 0.224, 0.225])[:, None, None] + np.array([0.485, 0.456, 0.406])[:, None, None])
        # batch_recon = np.clip(batch_recon * 255, 0, 255).astype(np.uint8)
        # img = Image.fromarray(np.transpose(batch_recon, (1, 2, 0)))
        # img.save(f'/cluster/home/abizeul/mae/tools/dct/batch_recon/image_{i}_{j}.png')

        # # # Apply filtering (zeroing out coefficients)
        print("this is the shape",order.shape,coef.shape)
        nb_dim = coef.shape[1] * coef.shape[2]
        for idx in range(0,nb_dim,12544):
            print(idx,idx+12544)
            zeros = np.zeros_like(coef)
            zeros.reshape([zeros.shape[0], -1])[:, order[idx:(idx+12544)]] = coef.reshape([zeros.shape[0], -1])[:, order[idx:(idx+12544)]]
            zeros = zeros.reshape(b.shape)
            filtered = np.stack([idct2(zeros[0]),idct2(zeros[1]),idct2(zeros[2])],0)

            # Save filtered images
            filtered = (filtered * np.array([0.229, 0.224, 0.225])[:, None, None] + np.array([0.485, 0.456, 0.406])[:, None, None])
            filtered = np.clip(filtered * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(np.transpose(filtered, (1, 2, 0)))
            img.save(f'/cluster/home/abizeul/mae/tools/dct/filtered/image_{i}_{j}_{idx}.png')

            # if idx>10: break

        # print("Size of the transformations:", batch.shape, coef.shape, batch_recon.shape)
        # print("Mean squared error:", mean_squared_error(b.flatten(), batch_recon.flatten()))

        # Break after a few iterations to avoid excessive image saving (optional)
        if i >= 2:
            break