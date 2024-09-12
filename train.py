import torch
import torch.nn as nn

from utils.data_loader import get_data_loaders

train_loader, val_loader = get_data_loaders()
print(train_loader)
print(val_loader)