import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define folder paths
images_folder = 'images'
unet_folder = 'Unet'
dlv3_folder = 'DLV3+'
comparison_folder = 'comparison'

# Create the comparison folder if it doesn't exist
if not os.path.exists(comparison_folder):
    os.makedirs(comparison_folder)

# Get list of image files from the "images" folder
image_files = [f for f in os.listdir(images_folder) if f.endswith('.png') or f.endswith('.jpg')]
image_files.sort()  # Sort for consistent ordering

# Process each image with a progress bar
for image_file in tqdm(image_files, desc="Generating comparison plots"):
    # Construct full paths to the original image and prediction masks
    original_path = os.path.join(images_folder, image_file)
    unet_path = os.path.join(unet_folder, image_file)
    dlv3_path = os.path.join(dlv3_folder, image_file)

    # Read the images using PIL
    original_img = Image.open(original_path)

    # Convert the image to a NumPy array for pixel value checking
    img_array = np.array(original_img)

    # Check if all pixels are black (RGB = 0, 0, 0 or grayscale = 0)
    if np.all(img_array == 0):
        continue

    unet_img = Image.open(unet_path)
    dlv3_img = Image.open(dlv3_path)

    # Create a figure with three subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(45, 15))

    # Display the original image
    axes[0].imshow(original_img)
    axes[0].set_title('Input Image', fontsize=36)  # Increased font size for subplot title
    axes[0].axis('off')

    # Display the U-Net prediction
    axes[1].imshow(unet_img)
    axes[1].set_title('UNet', fontsize=36)  # Increased font size for subplot title
    axes[1].axis('off')

    # Display the DeepLabV3+ prediction
    axes[2].imshow(dlv3_img)
    axes[2].set_title('DeeplabV3+', fontsize=36)  # Increased font size for subplot title
    axes[2].axis('off')

    # Set the overall title of the plot to the image file name with increased font size
    fig.suptitle(image_file, fontsize=30)

    # Reduce gaps between subplots
    plt.subplots_adjust(wspace=0.02)  # Adjust horizontal spacing (default is ~0.2)

    # Define the save path and save the plot
    save_path = os.path.join(comparison_folder, image_file)
    plt.savefig(save_path, bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(fig)