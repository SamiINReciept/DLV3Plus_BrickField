import os
from PIL import Image
import numpy as np


image = np.array(Image.open("dhaka_Kleene_2019_data_512/train/images/patch_000_000.png")).astype(np.uint8)

print(image.shape)