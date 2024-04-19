import numpy as np
import matplotlib.pyplot as plt
# import tifffile
import os
import random
# from scipy import ndimage
# transformer 4.30.0
from transformers import SamModel, SamProcessor
import torch

model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
data = torch.randint(0, 256, size=(2, 3, 200, 200), dtype=torch.float)
data_proc = processor(data,  return_tensors="pt")
data_proc = {k: v.to(device) for k, v in data_proc.items()}
output = model(data_proc['pixel_values'])
emb = model.get_image_embeddings(data_proc['pixel_values']) #torch.Size([2, 256, 64, 64])
predicted_masks = output.pred_masks.squeeze(1)
print(predicted_masks.shape) #torch.Size([2, 1, 256, 256])
predicted_masks_prob = torch.sigmoid(predicted_masks)
print(predicted_masks_prob.shape) #torch.Size([2, 1, 256, 256])
import pdb
pdb.set_trace()


