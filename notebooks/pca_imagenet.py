import os
import numpy as np
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from multiprocessing import Pool
from torchvision import transforms

# Path to the ImageNet dataset (with class folders)
dataset_folder = '/cluster/project/sachan/callen/data_alice'
dataset_dir = f'{dataset_folder}/ILSVRC2012_img/train'

# Image dimensions (ImageNet images are generally 224x224)
img_size = (224, 224)

# Batch size for IncrementalPCA
batch_size = 1024

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


# Define the transformations (replacing the resize)
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
    transforms.ToTensor(),              # Convert image to tensor
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize with ImageNet mean and std
])

# Function to preprocess a single image
def preprocess_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB')  # Ensure it's RGB
        img = transform(img)  # Apply the defined transforms
        return img.numpy().flatten()  # Convert to NumPy array and flatten
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# Load and preprocess images in parallel
def load_images_parallel(image_paths, num_workers=4):
    with Pool(num_workers) as p:
        images = list(tqdm(p.imap(preprocess_image, image_paths), total=len(image_paths)))
    return np.array([img for img in images if img is not None])

# Function to get all image paths from the dataset
def get_image_paths(dataset_dir):
    image_paths = []
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_name.endswith('.JPEG'):
                    image_paths.append(img_path)
    return image_paths

# Main function to run PCA on ImageNet dataset
def run_pca_on_imagenet(dataset_dir, n_components=100, batch_size=500, num_workers=4):
    # Get all image paths
    image_paths = get_image_paths(dataset_dir)
    total_images = len(image_paths)
    pca = IncrementalPCA(n_components=n_components)

    # Process images in batches
    for i in tqdm(range(0, total_images, batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        images = load_images_parallel(batch_paths, num_workers=num_workers)

        if images.shape[0] > 0:
            pca.partial_fit(images)

    return pca

# Run PCA on ImageNet dataset
pca_model = run_pca_on_imagenet(dataset_dir, batch_size=batch_size)

# np.save(f'{dataset_folder}/ILSVRC2012_img/imagenet_pc_matrix.npy',pca_model.components_)
np.save(f'{dataset_folder}/ILSVRC2012_img/imagenet_eigenvalues.npy',pca_model.explained_variance_)
np.save(f'{dataset_folder}/ILSVRC2012_img/imagenet_eigenvalues_ratio.npy',pca_model.explained_variance_ratio_)
