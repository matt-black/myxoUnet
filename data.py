"""Datasets
"""

# pytorch
import torch
from torch.utils.data import Dataset
# pillow
from PIL import Image

class BinaryDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.msk_dir = mask_dir
    
        
