from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = sys.maxsize
import os
from patch import create_image_patches
# from encode_labels_model_input import encode_bw_mask


# Get the parent directory of the current working directory
current_directory = os.getcwd()


# print(os.listdir(os.path.join(current_directory, "data/train/images"))[5])

# input_train = "Src/Dhaka.png"

# input_train = "Src/train_dhaka_input.tif"
# input_train_gt = "Src/train_dhaka_ground_truth.tif"

input_train = "data/klin/test_split/test_dhaka_input.tif"
input_train_gt = "data/klin/test_split/test_dhaka_ground_truth.tif"

# input_test = "Src/test_dhaka_input.tif"
# input_test_gt = "Src/test_dhaka_ground_truth.tif"

train_dir = os.path.join(current_directory, "output/test")
# test_dir = "data/test"
# val_dir = "data/val"

# create_image_patches(input_train, input_train_gt, train_dir, 512, 128, patch_function=None)
create_image_patches(input_train_gt, train_dir, 512, 512, extension="png", patch_function=None)

