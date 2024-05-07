import torch
import torch.nn as nn
from monai.networks.nets.vit import ViT
from monai.networks.nets import ViTAutoEnc

import pdb

model = ViTAutoEnc(
    in_channels=1,
    img_size=(512, 512, 48),
    patch_size=(16, 16, 16),
    pos_embed="conv",
    hidden_size=768,
    mlp_dim=3072,
    save_attn=True
).to('cuda')

pdb.set_trace()
img = torch.rand((1, 1, 512, 512, 48)).to('cuda')
out, hidden = model(img)
torch.stack(hidden)
