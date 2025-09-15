from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import rasterio
from rasterio.windows import Window


# Groks Corrected version (for leftover region alignment)
def create_image_patches(image_path, output_dir, patch_size=1500, stride=1500, extension="", patch_function=None, patch_name_suffix=None, label_path=None):
    """
    Divide a large GIS image (and optionally its label) into patches.

    Args:
        image_path (str): Path to the input GIS image.
        output_dir (str): Directory to save the patches.
        patch_size (int): Size of each patch (default: 1500).
        stride (int): Step between patches (default: 1500 for non-overlapping).
        label_path (str, optional): Path to the corresponding label image. If None, only process the image.
    """
    # Ensure output directories exist
    if not os.path.exists(output_dir+'/gts'):
        os.makedirs(output_dir+'/gts')
    
    if label_path:    
        if not os.path.exists(output_dir+'/gts'):
            os.makedirs(output_dir+'/gts')

    if not patch_name_suffix:
        patch_name_suffix = ""

    # Open the image and optionally the label
    with rasterio.open(image_path) as image:
        if label_path:
            with rasterio.open(label_path) as label:
                # Verify that image and label dimensions match
                assert image.width == label.width and image.height == label.height, \
                    f"Image and label sizes do not match: {image.width}x{image.height} != {label.width}x{label.height}"
                process_labels = True
        else:
            process_labels = False

        
        # Compute patch positions along height (y-axis)
        y_positions = list(range(0, image.height - patch_size + 1, stride))
        if y_positions and y_positions[-1] + patch_size < image.height:
            y_positions.append(image.height - patch_size)  # Include bottom edge

        # Compute patch positions along width (x-axis)
        x_positions = list(range(0, image.width - patch_size + 1, stride))
        if x_positions and x_positions[-1] + patch_size < image.width:
            x_positions.append(image.width - patch_size)  # Include right edge

        # Calculate total number of patches
        total = len(y_positions) * len(x_positions)
        print(f"Generating {total} patches: {len(y_positions)} rows x {len(x_positions)} columns")

        desc = "Creating patches (with labels)" if process_labels else "Creating patches (image only)"
        
        # Process patches with progress bar
        with tqdm(total=total, desc=desc) as pbar:
            for i, y in enumerate(y_positions):
                for j, x in enumerate(x_positions):
                    # Define the patch window
                    window = Window(x, y, patch_size, patch_size)

                    # Read and save the image patch
                    image_patch = image.read(window=window)
                    patch_transform = rasterio.windows.transform(window, image.transform)
                    patch_profile_image = image.profile.copy()
                    patch_profile_image.update({
                        "height": patch_size,
                        "width": patch_size,
                        "transform": patch_transform
                    })
                    image_filename = f"{output_dir}/images/patch_{i:03d}_{j:03d}.{extension}"
                    with rasterio.open(image_filename, 'w', **patch_profile_image) as dst_image:
                        dst_image.write(image_patch)

                    # If labels are provided, read and save the label patch
                    if process_labels:
                        label_patch = label.read(window=window)
                        patch_profile_label = label.profile.copy()
                        patch_profile_label.update({
                            "height": patch_size,
                            "width": patch_size,
                            "transform": patch_transform
                        })
                        label_filename = f"{output_dir}/gts/patch_{i:03d}_{j:03d}_gt.{extension}"
                        with rasterio.open(label_filename, 'w', **patch_profile_label) as dst_label:
                            dst_label.write(label_patch)

                    pbar.update(1)






# def create_image_patches(image_path, label_path, output_dir, patch_size=512, stride=256, patch_name_suffix=None, patch_function=None):
#     """
#     Takes an image file located at `image_path` and label file `label_path` and creates patches of size `patch_size` with a stride of `stride`.
#     Applies the optional `patch_function` to label patch and saves the resulting patches in `output_dir`. Besides The function
#     also generates a image patch in `output_dir`.

#     Args:
#         image_path (str): The path to the input image file.
#         label_path (str): The path to the label image file.
#         output_dir (str): The directory where the output patches and label file will be saved.
#         patch_size (int , optional): The size of each patch. If an integer is provided, the patch is assumed
#             to be square with sides of length `patch_size`.
#             dimensions of the patch. Default is 512.
#         stride (int , optional): The stride between consecutive patches. If an integer is provided, the stride
#             is assumed to be the same in both dimensions.
#             stride between consecutive patches. Default is 256.
#         patch_name_suffix (str, optional): A string to append to the end of each patch's filename. Default is None.
#         patch_function (function, optional): A function to apply to each patch before saving. The function should take
#             a single argument (the patch) and return the modified patch. Default is None.

#     Raises:

#     Returns:
#         None

#     To-do:
#     # - zeroing image by background label
#     # - Remove full bg patches with image patch
#     - add edge label augmentation
#     """
#     assert os.path.isfile(image_path), f"Invalid image file: {image_path}"
#     assert os.path.isfile(label_path), f"Invalid label file: {label_path}"
#     assert isinstance(patch_size, int) and patch_size > 0, "patch_size should be an integer."
#     assert isinstance(stride, int) and stride > 0, "stride should be a valid integer."
#     assert patch_function is None or callable(patch_function), "patch_function should be a callable function."
    

#     if not os.path.exists(output_dir+'/images'):
#         os.makedirs(output_dir+'/images')
    
#     if not os.path.exists(output_dir+'/gts'):
#         os.makedirs(output_dir+'/gts')


#     if not patch_name_suffix:
#         patch_name_suffix = ""


#     # Marking beginning of changes

#     with rasterio.open(image_path) as image, rasterio.open(label_path) as label:
#         assert image.width == label.width and image.height == label.height, f"Image and label sizes do not match: {image.width}x{image.height} != {label.width}x{label.height}"

#     # image = Image.open(image_path).convert("RGB")
#     # label = Image.open(label_path).convert("RGB")

#     # assert len(image.mode) == 3, f"Invalid image mode {image.mode}"
#     # assert len(label.mode) == 3, f"Invalid label mode {label.mode}"
#     # assert image.size == label.size, f"Both image and label have to be same size but image: {image.size} and label: {label.size}"

#         total = len(range(0, image.height - patch_size + 1, stride)) * len(range(0, image.width - patch_size + 1, stride))
#         last_i = 0
#         last_y = 0
#         leftover_height = 0


#         with tqdm(total= total) as pbar:
#             # Loop over the image and create patches
#             for i, y in enumerate(range(0, image.height - patch_size + 1, stride)):
#                 for j, x in enumerate(range(0, image.width - patch_size + 1, stride)):

#                     # print(f"Height Point = {y}, Width Point = {x}")

#                     # Define the window for cropping
#                     window = Window(x, y, patch_size, patch_size)

#                     # Read the image and label patches
#                     image_patch = image.read(window=window)
#                     label_patch = label.read(window=window)
                    
#                     # left = x
#                     # top = y
#                     # right = x+patch_size
#                     # bot = y+patch_size

#                     # if right > image.width:
#                     #     x = right - image.width
#                     #     right = image.width

#                     # if bot > image.height:
#                     #     y = bot - image.height
#                     #     bot = image.height

#                     # # Crop the patch
#                     # image_patch = image.crop((left, top, right, bot))
#                     # label_patch = label.crop((left, top, right, bot))

#                     # image_patch_array = np.array(image_patch)
#                     # label_patch_array = np.array(label_patch)

#                     # skip unnecessary background patch
#                     # if np.sum(image_patch_array) == 0 or np.sum(label_patch_array) == 0:
#                     #     pbar.update(1)
#                     #     continue

#                     # Skip unnecessary background patches
#                     # if np.sum(image_patch) == 0 or np.sum(label_patch) == 0:
#                     #     pbar.update(1)
#                     #     continue

#                     # zeroing image by background label
#                     # mask = (label_patch_array == [0, 0, 0]).all(axis=2)
#                     # image_patch_array[mask] = [0, 0, 0]

#                     # image_patch = Image.fromarray(image_patch_array)
#                     # label_patch = Image.fromarray(label_patch_array)

#                     # if patch_function:
#                     #     label_patch = patch_function(label_patch)

#                     # Update metadata for each patch
#                     patch_transform = rasterio.windows.transform(window, image.transform)
#                     patch_profile_image = image.profile.copy()
#                     patch_profile_label = label.profile.copy()

#                     patch_profile_image.update({
#                         "height": patch_size,
#                         "width": patch_size,
#                         "transform": patch_transform
#                     })
#                     patch_profile_label.update({
#                         "height": patch_size,
#                         "width": patch_size,
#                         "transform": patch_transform
#                     })

#                     # Save the patch to a file
                    
#                     image_filename = f"{output_dir}/images/patch_{i:03d}_{j:03d}.tif"
#                     label_filename = f"{output_dir}/gts/patch_{i:03d}_{j:03d}_gt.tif"
                    
#                     # image_patch.save(image_filename, format='TIFF', compress_level=0, optimize=False)
#                     # label_patch.save(label_filename, format='TIFF', compress_level=0, optimize=False)

#                     with rasterio.open(image_filename, 'w', **patch_profile_image) as dst_image:
#                         dst_image.write(image_patch)

#                     with rasterio.open(label_filename, 'w', **patch_profile_label) as dst_label:
#                         dst_label.write(label_patch)
                    
#                     last_i = i
#                     last_y = y
#                     leftover_height = image.height - (y+patch_size)

#                     pbar.update(1)

#         total = len(range(0, image.width - patch_size + 1, stride))
#         with tqdm(total= total) as pbar:
#             for j, x in enumerate(range(0, image.width - patch_size + 1, stride)):
#                 # Define the window for cropping
#                     i = last_i+1
#                     y = last_y + leftover_height
#                     print(f"Height Point = {y}, Width Point = {x}")

#                     window = Window(x, y, patch_size, patch_size)

#                     # Read the image and label patches
#                     image_patch = image.read(window=window)
#                     label_patch = label.read(window=window)

#                     # Update metadata for each patch
#                     patch_transform = rasterio.windows.transform(window, image.transform)
#                     patch_profile_image = image.profile.copy()
#                     patch_profile_label = label.profile.copy()

#                     patch_profile_image.update({
#                         "height": patch_size,
#                         "width": patch_size,
#                         "transform": patch_transform
#                     })
#                     patch_profile_label.update({
#                         "height": patch_size,
#                         "width": patch_size,
#                         "transform": patch_transform
#                     })

#                     # Save the patch to a file
#                     image_filename = f"{output_dir}/images/patch_{i:03d}_{j:03d}.tif"
#                     label_filename = f"{output_dir}/gts/patch_{i:03d}_{j:03d}_gt.tif"
                    
#                     with rasterio.open(image_filename, 'w', **patch_profile_image) as dst_image:
#                         dst_image.write(image_patch)
#                     with rasterio.open(label_filename, 'w', **patch_profile_label) as dst_label:
#                         dst_label.write(label_patch)

#                     pbar.update(1)

