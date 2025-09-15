import os
import numpy as np
import rasterio as rio
from PIL import Image
from tqdm import tqdm

def convert_to_black(image_path, save_path, join_str):
    save_path = save_path+'/blackend_annotations'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Open the TIFF image
    with Image.open(image_path) as img:
        # Convert the image to RGB if it's not already in RGB mode
        img = img.convert("RGB")

        # Create a new black image with the same size
        black_img = Image.new("RGB", img.size, (0, 0, 0))

        # Copy the metadata (if available) from the original image
        # black_img.info = img.info
        # print(img.info)

        # Save the new black image
        black_img.save(os.path.join(save_path, join_str),  format="PNG")

def blacken(dir_path):
    img_path = os.path.join(dir_path, "images_patches")
    annot_path = os.path.join(dir_path, "geo_annotations")

    images = os.listdir(img_path)
    annotations = os.listdir(annot_path)
    non_annot_images = []
    
    for name in images:
        if name not in annotations:
            non_annot_images.append(name)
    
    non_annot_index = 0
    with tqdm(total= len(images)) as pbar:
        for i in range(len(images)):
            if(images[i] == non_annot_images[non_annot_index]):
                convert_to_black(os.path.join(img_path, images[i]), dir_path, non_annot_images[non_annot_index])
                non_annot_index += 1
            pbar.update(1)

    # original_tif_path = os.path.join(tif_path, os.listdir(tif_path)[i])


# blacken(os.path.join(os.getcwd(), "data/train"))
blacken(os.path.join(os.getcwd(), "data/klin"))
