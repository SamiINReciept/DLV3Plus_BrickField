import os
import argparse
from tqdm import tqdm  # Import tqdm for progress bar
# from mainDLV3Plus_testUnseen import main  # Replace 'your_script' with the actual filename of your code
from mainDLV3Plus import main  # Replace 'your_script' with the actual filename of your code

def run_for_all_subfolders(root_dir):
    # Ensure the root directory exists
    if not os.path.exists(root_dir):
        print(f"Root directory '{root_dir}' does not exist!")
        return

    # Get list of subfolders
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    print(f"Found {len(subfolders)} subfolders in '{root_dir}'")

    # Base arguments that won't change
    base_args = {
        "batch_size": 128,
        "epochs": 30,
        "lr": 0.001,
        "num_classes": 2,
        "checkpoint_path": "checkpoint.pth",
        "best_model_path": "checkpoint.pth",
        "losses_path": "checkpoints/losses.pth",
        "eval_only": True,  # Since you're predicting, keep this True
        "restart": False,
        "run_id": None
    }

    # Iterate over each subfolder with tqdm progress bar
    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        subfolder_path = os.path.join(root_dir, subfolder)
        print(f"Processing subfolder: {subfolder_path}")

        # Create an argument object for this subfolder
        parser = argparse.ArgumentParser(description="Driver script for running model on multiple subfolders")
        for key, value in base_args.items():
            parser.add_argument(f"--{key}", default=value, type=type(value) if value is not None else str)
        parser.add_argument("--datadir", type=str, default=subfolder_path, help="Path to the dataset directory")

        # Parse arguments for this subfolder
        args = parser.parse_args([])  # Empty list since we're not using command-line input here
        args.datadir = subfolder_path  # Explicitly set datadir to the current subfolder

        # Call your main function with these arguments
        try:
            main(args)
            print(f"Completed processing for {subfolder_path}")
        except Exception as e:
            print(f"Error processing {subfolder_path}: {e}")

if __name__ == "__main__":
    root_directory = "/home/temp/ten_dist_2019_patched"
    run_for_all_subfolders(root_directory)