import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import sys
sys.path.append("/cluster/home/callen/projects/mae")
from utils import Normalize
import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from torchvision.transforms import v2


mean = torch.tensor(np.reshape(np.load('/cluster/project/sachan/callen/data_alice/CLEVR_v1.0/images/clevr_mean.npy'),[3,224,224]))
std = torch.tensor(np.reshape(np.load('/cluster/project/sachan/callen/data_alice/CLEVR_v1.0/images/clevr_std.npy'),[3,224,224]))

mask_tf = v2.Compose([
    v2.PILToTensor(),
    v2.Resize(size=[224, 224]),
    v2.ToTensor()
])
img_tf = v2.Compose([
    # v2.PILToTensor(),
    v2.Resize(size=[224, 224]),
    v2.ToTensor(),
    Normalize(mean,std)
])

L = 100000

def segmask_to_box(m):
    box_mask = torch.zeros_like(m)
    if m.sum() != 0:
        idx_y = (m.sum(0) != 0).nonzero()
        idx_x = (m.sum(1) != 0).nonzero()
        x1, x2 = idx_x[0].item(), idx_x[-1].item()
        y1, y2 = idx_y[0].item(), idx_y[-1].item()
        box_mask[x1:x2, y1:y2] = 1
    else:
        box_mask = m
    return box_mask


class CLEVRCustomDataset(Dataset):
    def __init__(self, root, transform=None, split='train', morph='none', load_masks=False):
        self.data_dir = f"{root}/clevr"
        self.split = split
        self.transform = transform

        if not os.path.exists(f"{self.data_dir}/train_indices.pt"):
            indices = torch.randperm(L)
            torch.save(indices[:int(L * 0.7)], f"{self.data_dir}/train_indices.pt")
            torch.save(indices[int(L * 0.7):], f"{self.data_dir}/val_indices.pt")

        self.split_indices = torch.load(f"{self.data_dir}/{split}_indices.pt")
        shape = torch.load(f"{self.data_dir}/shape.pt")
        color = torch.load(f"{self.data_dir}/color.pt")
        self.color, self.shape = color[self.split_indices], shape[self.split_indices]

        self.C, self.S = len(self.color.unique())-1, len(self.shape.unique())-1
        self.n_objects = (self.color!=0).sum(-1)

        self.load_masks = load_masks
        kernel = np.ones((10, 10))
        if morph == "none":
            self.morph_fn = lambda x: x
        elif morph == "erosion":
            self.morph_fn = lambda x: torch.tensor(binary_erosion(x, structure=kernel)).float()
        elif morph == "dilation":
            self.morph_fn = lambda x: torch.tensor(binary_dilation(x, structure=kernel)).float()
        elif morph == "box":
            self.morph_fn = lambda m: segmask_to_box(m)

    def __len__(self):
        return len(self.split_indices)

    def seg_to_binary_mask(self, idx, view=0):
        masks = []
        for i in range(11):
            mp = f"{self.data_dir}/masks/{self.split_indices[idx]}_{i}.png"
            m = mask_tf(Image.open(mp))
            mask = torch.zeros_like(m)
            mask[m > 0.5] = 1.
            masks.append(self.morph_fn(mask))
        final_mask = torch.stack(masks)[1:, ...]
        return final_mask

    def __getitem__(self, idx):
        imgpath = f"{self.data_dir}/images/{self.split_indices[idx]}.png"
        img = Image.open(imgpath)
        anno = {}

        img = img_tf(img).to(dtype=torch.float32)

        shape, color = self.shape[idx], self.color[idx]
        shape, color = shape[shape!=0], color[color!=0]
        label = color + self.C*(shape-1) -1
        one_hot = torch.zeros(self.C * self.S)
        one_hot[label.long()] = 1.
        anno['labels'] = one_hot

        if self.load_masks:
            if self.split == 'train':
                anno_masks = []
                anno_masks.append(self.seg_to_binary_mask(idx, 0))
                anno['mask'] = torch.stack(anno_masks, 1)
            else:
                anno['mask'] = self.seg_to_binary_mask(idx)
            anno['n_objects'] = self.n_objects[idx]

        if self.split == "train":
            return img, [anno["labels"], anno["mask"].squeeze(1)]
        else:
            return img, anno["labels"]

if __name__ == "__main__":
    dir_data = "/usr/scratch/data"
    load_masks = True
    train_dst = CLEVRCustomDataset(dir_data, load_masks=load_masks)
