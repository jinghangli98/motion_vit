import torch
import monai
import sys
import nibabel as nib
import pdb
from torchvision.transforms import CenterCrop

input = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=5)
model.load_state_dict(torch.load('./checkpoint/weight_20.pth'))
model = model.to(device)
transform = CenterCrop((512, 512,))
imgs = torch.tensor(nib.load(input).get_fdata()).permute(-1,0,1).to(device).float()
ratings = [model(transform(img).unsqueeze(0).unsqueeze(0)).softmax(dim=1).argmax().detach().cpu() for img in imgs]
rating = torch.stack(ratings).float().mean()

print(f'Input: {input} | Motion Rating: {rating.item()}')
