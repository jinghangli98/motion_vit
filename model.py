import torch
import torch.nn as nn
from monai.networks.nets.vit import ViT
from monai.networks.nets import ViTAutoEnc
import monai
import pdb

# model = ViTAutoEnc(
#     in_channels=1,
#     img_size=(464, 464, 48),
#     patch_size=(16, 16, 16),
#     pos_embed="conv",
#     hidden_size=768,
#     mlp_dim=3072,
#     save_attn=True
# )
# model = ViT(in_channels=1, 
#             img_size=(464, 464, 48), 
#             patch_size=(16, 16, 16), 
#             proj_type='conv', 
#             pos_embed_type='sincos',
#             save_attn=True,
#             classification=True,
#             num_classes = 5)
# model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=5).to('cuda')
model = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=5).to('cuda')
# model = monai.networks.nets.TorchVisionFCModel(model_name='resnet18', num_classes=5, dim=2, in_channels=None)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False).to('cuda')
# original_first_layer = model.conv1
# model.conv1 = torch.nn.Conv2d(1, original_first_layer.out_channels,
#                               kernel_size=original_first_layer.kernel_size, 
#                               stride=original_first_layer.stride, 
#                               padding=original_first_layer.padding, 
#                               bias=original_first_layer.bias)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 5)

# model = model.to('cuda')
# img = torch.rand((1, 1, 512, 512, 48)).to('cuda')
# out, hidden = model(img)
# pdb.set_trace()
# torch.stack(hidden)
