#!/bin/bash

# Define variables
SOURCE_DIR="/media/ccds/Local Disk/Downloads/district_patches"
REMOTE_USER="temp"
REMOTE_HOST="103.159.2.125"
REMOTE_PATH="/home/temp/Sami_Unet_BrickField/Unet_BrickField/64Dist_Preds_Remaining"
PASSWORD="34512312"  # Replace with your actual password

# Check if sshpass is installed, install it if not (assuming Ubuntu/Debian-based WSL)
if ! command -v sshpass &> /dev/null; then
    echo "sshpass not found. Installing it..."
    sudo apt update && sudo apt install -y sshpass
fi

# Check if SOURCE_DIR exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist."
    exit 1
fi

# Loop through each directory in SOURCE_DIR and upload it
for dir in "$SOURCE_DIR"/*/ ; do
    if [ -d "$dir" ]; then  # Check if it's a directory
        dir_name=$(basename "$dir")
        echo "Uploading directory: $dir_name"
        
        # Use sshpass to provide the password to rsync over SSH
        sshpass -p "$PASSWORD" rsync -rz --info=progress2 --info=name0 -e ssh "$dir" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$dir_name"
        
        if [ $? -eq 0 ]; then
            echo "Successfully uploaded $dir_name"
        else
            echo "Error uploading $dir_name"
        fi
    else
        echo "No directories found in $SOURCE_DIR"
    fi
done

echo "Upload process completed."
