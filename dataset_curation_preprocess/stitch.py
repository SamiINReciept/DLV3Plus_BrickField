import os
from PIL import Image
from tqdm import tqdm

def stitch_images(patch_dir, output_path, rows, cols):
    # Get all TIFF image patches from the directory
    patch_files = [f for f in os.listdir(patch_dir)]
    # if not patch_files:
    #     print("No TIFF images found in the directory.")
    #     return

    # Open the first image to get its dimensions (assuming all patches are the same size)
    first_patch = Image.open(os.path.join(patch_dir, patch_files[0]))
    patch_width, patch_height = first_patch.size
    
    # Create a blank canvas for the final stitched image
    stitched_width = patch_width * cols
    stitched_height = patch_height * rows
    stitched_image = Image.new('RGB', (stitched_width, stitched_height))

    # Place each patch on the canvas at the correct position
    with tqdm(total = len(patch_files)) as pbar:
        for i, patch_file in enumerate(patch_files):
            # Calculate the position of the current patch (row and column in the grid)
            row = i // cols
            col = i % cols
            x_offset = col * patch_width
            y_offset = row * patch_height
            
            # Open the patch image
            patch = Image.open(os.path.join(patch_dir, patch_file))
            
            # Paste the patch on the final image
            stitched_image.paste(patch, (x_offset, y_offset))

            pbar.update(1)

    # Save the stitched image
    stitched_image.save(os.path.join(output_path, "stiched.tif"), format="TIFF")
    print(f"Stitched image saved at {output_path}")


# patch_dir = os.path.join(os.getcwd(), "data/train/all_annotations")  # Directory with TIFF patches
# output_path = os.path.join(os.getcwd(), "data/train/stiched_annotation")  # Output path for the stitched image

patch_dir = os.path.join(os.getcwd(), "data/klin/gt_patches")  # Directory with TIFF patches
output_path = os.path.join(os.getcwd(), "data/klin/stiched_annotation")  # Output path for the stitched image

rows = 34  # Number of rows in the grid
cols = 9  # Number of columns in the grid

stitch_images(patch_dir, output_path, rows, cols)
