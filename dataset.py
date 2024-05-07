import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torchio as tio

class tse_dataset(Dataset):
    def __init__(self, nii_dir, transform=None):
        self.nii_dir = nii_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.nii_dir)
    
    def __getitem__(self, idx):
        
        pt = torch.load(self.nii_dir[idx])
        img = pt['tse']
        rating = pt['rating']
        
        return img, rating
    
dataset = tse_dataset(glob.glob('data/*.pt'))

