import os
import rasterio as rio
# from rasterio.enums import Resampling
import numpy as np
from PIL import Image
from tqdm import tqdm

#annotations are in data/train/annotations
def convert_png_to_tif(tif_path, mask_path, output_path):
    '''
    Takes an image(.TIF) file located at 'tif_path' and an annotation(.PNG) file located at 'mask_path', based on which generates new annotation(.TIF) files (with geolocations and other metadata) at 'output_path'.


    Args:
        tif_path (str): Path to the patched images, eg "data/train/images"
        #mask_path (str): Path to the annotations, eg "data/train/annotations"
        #output_path (str): Path to the geolocation embedded annotations, eg "data/train/geo_annotations"

    '''

    parent_path = os.path.dirname(tif_path) #parent folder: BrickField/data/train

    #creates the folder 'geo_annotations' inside 'BrickField/data/train'
    if not os.path.exists(output_path):
        os.makedirs(parent_path+'/geo_annotations')
        # os.makedirs(parent_path+'/geo_blackend_annotations')

    tif_count = len(os.listdir(tif_path))   # counts the number of patches (.tif)
    mask_count = len(os.listdir(mask_path)) # counts the number of annotations (.png)
    total = tif_count   # used for progress bar in tqdm

    #verifies that there is equal amount of annotations for the number of images available
    # assert tif_count == mask_count, f"Count mismatch: Images and Annotations count does not match, one or more annotations might be missing"

    mask_index = 0
    with tqdm(total= total) as pbar:
        for i in range(total):
            if(mask_index==mask_count):
                break

            if(os.listdir(tif_path)[i][:13] != os.listdir(mask_path)[mask_index][:13]):
                # os.listdir(tif_path)[i]
                # os.listdir(mask_path)[i]cl
                continue

            original_tif_path = os.path.join(tif_path, os.listdir(tif_path)[i]) # Filename: "data/train/images/patch_000_000.tif"
            
            # masked_png_path = os.path.join(mask_path, os.listdir(mask_path)[i])
            masked_png_path = os.path.join(mask_path, os.listdir(mask_path)[mask_index])
            
            new_tif_path = f"{output_path}/{os.listdir(tif_path)[i]}"    # Filename: "data/train/geo_annotations/masked_patch_025_007.tif"

            # Open the TIFF file to extract metadata
            with rio.open(original_tif_path) as tif:
                metadata = tif.meta.copy()  # Copy metadata (CRS, transform, etc.)

            # Open the PNG file to get visual data
            png_image = Image.open(masked_png_path)
            png_array = np.array(png_image)  # Convert PNG to a numpy array

            # Ensure PNG data matches the TIFF dimensions
            if png_array.shape[:2] != (metadata['height'], metadata['width']):
                print("Error: dimension mismatch")
                # Resize the PNG to match the TIFF dimensions
                png_image_resized = png_image.resize((metadata['width'], metadata['height']), Image.ANTIALIAS)
                png_array = np.array(png_image_resized)

            # Prepare the data for writing (ensure correct number of bands)
            if len(png_array.shape) == 2:  # Grayscale PNG
                png_array = np.expand_dims(png_array, axis=0)  # Add a single band
            else:  # RGB PNG
                png_array = np.moveaxis(png_array, -1, 0)  # Move channels to first axis

            # Update metadata for the new TIFF
            metadata.update({
                "count": png_array.shape[0],  # Number of bands
                "dtype": png_array.dtype  # Data type of the PNG array
            })

            # Write the new TIFF file with PNG visual data and TIFF metadata
            with rio.open(new_tif_path, "w", **metadata) as dst:
                dst.write(png_array)

            mask_index += 1

            pbar.update(1)


