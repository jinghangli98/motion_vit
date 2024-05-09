import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchio as tio
import glob
import pdb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.transforms import CenterCrop

class tse_dataset(Dataset):
    def __init__(self, nii_dir, transform=None):
        self.nii_dir = nii_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.nii_dir)
    
    def __getitem__(self, idx):        
        pt = torch.load(self.nii_dir[idx])
        img = self.transform(torch.tensor(pt['slice']))
        rating = pt['rating']
        
        return img/img.max(), rating

        
        
        
transform = CenterCrop((512, 512,))

train_data, val_data = train_test_split(glob.glob('data/2d/*.pt'), test_size=0.2, random_state=42)
train_dataset = tse_dataset(train_data, transform)
val_dataset = tse_dataset(val_data, transform)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size = 16, shuffle=False)
