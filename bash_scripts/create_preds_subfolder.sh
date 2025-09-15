#!/bin/bash

# Define the root directory
ROOT_DIR="/media/ccds/Local Disk/Downloads/district_patches"

# Check if the root directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Root directory $ROOT_DIR does not exist."
    exit 1
fi

# Loop through each directory in ROOT_DIR and create a 'preds' folder
for dir in "$ROOT_DIR"/*/ ; do
    if [ -d "$dir" ]; then  # Check if it's a directory
        dir_name=$(basename "$dir")
        preds_path="$dir/preds"
        
        if [ ! -d "$preds_path" ]; then
            echo "Creating 'preds' folder in $dir_name"
            mkdir "$preds_path"
            if [ $? -eq 0 ]; then
                echo "Successfully created 'preds' in $dir_name"
            else
                echo "Error creating 'preds' in $dir_name"
            fi
        else
            echo "'preds' folder already exists in $dir_name"
        fi
    else
        echo "No directories found in $ROOT_DIR"
    fi
done

echo "Process completed."
