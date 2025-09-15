import os
from PIL import Image
from tqdm import tqdm

"""
    This code was used to Create all Files in png_11Class (all patches inside train and test in that folder)

    Stride determines how much overlap would be there between each patch 
    
    Code just Converts Tiff images to PNG images

"""

# Directories
image_tif_dir = "output/images"
gt_tif_dir = "output/gts_label"

# Output PNG directories (creating if not exists)
image_png_dir = "output\images_png"
gt_png_dir = "output/gts_label_png"

os.makedirs(image_png_dir, exist_ok=True)
os.makedirs(gt_png_dir, exist_ok=True)

def convert_tif_to_png(input_dir, output_dir):
    """
    Converts all TIFF images in `input_dir` to PNG format and saves them in `output_dir`.
    """
    tif_files = [f for f in os.listdir(input_dir) if f.endswith(".tif")]

    with tqdm(total=len(tif_files), desc=f"Converting {input_dir} to PNG", unit="file") as pbar:
        for tif_file in tif_files:
            tif_path = os.path.join(input_dir, tif_file)
            png_path = os.path.join(output_dir, tif_file.replace(".tif", ".png"))

            # Open TIFF and convert to PNG
            with Image.open(tif_path) as img:
                img.save(png_path, format="PNG", compress_level=0, optimize=False)  # No compression

            pbar.update(1)  # Update progress bar

# Convert both image and ground truth TIFFs to PNG
convert_tif_to_png(image_tif_dir, image_png_dir)
convert_tif_to_png(gt_tif_dir, gt_png_dir)

print("\n✅ Conversion completed! PNG images saved while keeping TIFF files intact.")
