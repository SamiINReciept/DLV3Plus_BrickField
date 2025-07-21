import os
import shutil

# def clear_preds_folders(root_dir="64Dist_preds"):
def clear_preds_folders(root_dir="64zip"):
    # Check if root directory exists
    if not os.path.exists(root_dir):
        print(f"Root directory '{root_dir}' not found!")
        return
    
    # Iterate through all subfolders in root directory
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # preds_path = os.path.join(subfolder_path, "preds")
            preds_path = os.path.join(subfolder_path, "images")
            
            # Check if preds folder exists
            if os.path.exists(preds_path) and os.path.isdir(preds_path):
                try:
                    # Remove all contents of preds folder
                    for item in os.listdir(preds_path):
                        item_path = os.path.join(preds_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)  # Remove file
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)  # Remove directory and its contents
                    print(f"Cleared contents of: {preds_path}")
                except Exception as e:
                    print(f"Error clearing {preds_path}: {str(e)}")
            else:
                print(f"No 'preds' folder found in: {subfolder_path}")

if __name__ == "__main__":
    # You can change the root directory path here if needed
    # clear_preds_folders("64Dist_Preds")
    clear_preds_folders("64zip")