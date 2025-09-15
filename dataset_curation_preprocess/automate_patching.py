import os
from patch import create_image_patches  # Import your patch creation function

def process_all_districts(source_dir, output_base_dir, patch_size=512, stride=512, extension="png"):
    """
    Process all district images and create patches with the specified folder structure.
    
    Args:
        source_dir (str): Directory containing original district images
        output_base_dir (str): Base directory where district folders will be created
        patch_size (int): Size of each patch (default: 512)
        stride (int): Step between patches (default: 512)
        extension (str): File extension for output patches (default: "png")
    """
    # Ensure output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # List of district image files
    district_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.tif'))]
    
    for filename in district_images:
        # Extract district name from filename (remove extension)
        district_name = os.path.splitext(filename)[0]
        
        # Create district folder and images subfolder
        district_dir = os.path.join(output_base_dir, district_name)
        images_dir = os.path.join(district_dir, 'images')
        
        # Full path to input image
        image_path = os.path.join(source_dir, filename)
        
        print(f"Processing {district_name}...")
        
        try:
            # Create patches using your function
            create_image_patches(
                image_path=image_path,
                output_dir=images_dir,
                patch_size=patch_size,
                stride=stride,
                extension=extension,
                patch_function=None,
                patch_name_suffix=None
            )
            print(f"Completed processing {district_name}")
            
        except Exception as e:
            print(f"Error processing {district_name}: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Specify your directories based on the provided path
    SOURCE_DIR = r"E:\Downloads\Base_patches"  # Replace with actual path from your screenshot
    OUTPUT_DIR = r"E:\Downloads\district_patches"  # Output directory for patched images
    
    process_all_districts(SOURCE_DIR, OUTPUT_DIR, patch_size=512, stride=512, extension="png")