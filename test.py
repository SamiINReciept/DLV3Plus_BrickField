# Importing the modules from the UNet folder
from DeepLabV3Plus.network.modeling import deeplabv3plus_resnet101
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms

import segmentation_models_pytorch as smp
from torchsummary import summary
from sklearn.metrics import f1_score

from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import logging
import argparse
import wandb

from DeepLabV3Plus.metrics import StreamSegMetrics
import matplotlib.pyplot as plt

model = torch.load("checkpoints/checkpoint.pth")

print(model.keys())
# print(model['epoch'])