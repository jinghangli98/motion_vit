import glob
import torch
from natsort import natsorted
import pdb
from tqdm import tqdm

files = natsorted(glob.glob('./data/3d/*.pt'))

for idx, file in enumerate(tqdm(files)):
    pt = torch.load(file)
    img = pt['tse']
    rating = pt['rating']
    for idx in range(img.shape[-1]):
        img_slice = img[:,:,idx]
        filename = file.split('/')[-1].split('.')[0] + f'_2d_{idx}.pt'
        torch.save({'slice':img_slice, 'rating':rating}, f'data/2d/{filename}')
        
    