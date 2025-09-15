from PIL import Image
import numpy as np
import os
from tqdm import tqdm

"""
    This code Encodes the (11 classes labels RGB) to (6 classes labels Numerical Labels as in each point of a pixel is represented by 1 class index value of
    the class they are representing rather than an RGB value) but they lose their color.

"""


# Define mapping from original 11-class labels to new 6-class labels
# class_map = {(0, 0, 0):      0,  # bg                   black
#             (0, 255, 0):     1,  # farmland             lime green
#             (0, 0, 255):     2,  # water                blue
#             (0, 255, 255):   3,  # forest               cyan
#             (128, 0, 0):     4,  # urban_structure      maroon     dark red
#             (255, 0, 255):   4,  # rural_built_up       magenta
#             (255, 0, 0):     4,  # urban_built_up       red
#             (160, 160, 164): 4,  # road                 grey
#             (255, 255, 0):   5,  # meadow               yellow
#             (255, 251, 240): 5,  # marshland            ivory       white
#             (128, 0, 128):   4  # brick_factory         purple
#             }

class_map = {(0, 0, 0):      0,  # non-brickfield       black
            (202, 99, 64):   1  # brickfield            orange
            }

def encode_bw_mask(rgb_mask, class_map=class_map):
    """
    Encode a PIL Image object containing an RGB mask into a binary mask with class labels.

    Parameters: 
    rgb_mask (PIL Image): An image object containing the RGB mask.
    class_map (dict): A dictionary mapping RGB values to class labels.

    Returns: 
    bw_mask (PIL Image): A binary mask with class labels.
    """

    # print("entering")
    # assert isinstance(rgb_mask, Image.Image), "Object is not a PIL Image"
    # assert all(isinstance(k, tuple) and len(k) == 3 for k in class_map.keys()), "Invalid keys in class_map"
    # assert all(isinstance(v, int) for v in class_map.values()), "Invalid values in class_map"

    rgb_mask_array = np.array(rgb_mask)
    # num_classes = len(set(class_map.values()))
    # channels = 3

    # assert rgb_mask_array.shape[2] == channels, f"Mask should have 3 channels but found {rgb_mask_array.shape[2]}."
    # assert len(np.unique(rgb_mask_array)) <= num_classes * channels, "RGB mask has more classes than expected"

    # Label encode the mask
    bw_mask = np.zeros((rgb_mask_array.shape[0], rgb_mask_array.shape[1]), dtype=np.uint8)
    for rgb_val, class_label in class_map.items():
        indices = np.where(np.all(rgb_mask_array == rgb_val, axis=-1))
        bw_mask[indices] = class_label

    # assert len(bw_mask.shape) == 2, f"Invalid Shape {bw_mask.shape}"
    # assert len(np.unique(bw_mask)) <= num_classes, f"Invalid number of classes {len(np.unique(bw_mask))}"

    bw_mask = Image.fromarray(bw_mask)
    return bw_mask

# Processing masks in a directory
input_mask_dir = "output/test/gts"  # Folder containing original 11-class masks
output_mask_dir = "output/test/gts_label"  # Folder for new 6-class masks

os.makedirs(output_mask_dir, exist_ok=True)

# Process each mask
for filename in tqdm(os.listdir(input_mask_dir)):
    if filename.endswith(".png"):  # Ensure only PNG files are processed
        mask_path = os.path.join(input_mask_dir, filename)

        # Load RGB mask
        rgb_mask = Image.open(mask_path).convert("RGB")

        # Convert RGB mask to labeled grayscale mask
        label_mask = encode_bw_mask(rgb_mask)

        # Save the new label mask
        output_label_path = os.path.join(output_mask_dir, filename)
        label_mask.save(output_label_path)

# print("✅ Class merging completed! New masks saved in:", output_mask_dir)
