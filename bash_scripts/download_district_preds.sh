#!/bin/bash

# Define variables
LOCAL_DIR="/mnt/e/Downloads/64_Dist_Preds"
REMOTE_USER="temp"
REMOTE_HOST="103.159.2.125"
REMOTE_PATH="/home/temp/Sami_Unet_BrickField/Unet_BrickField/64Dist_Preds"
PASSWORD="34512312"  # Replace with your actual password

# Check if sshpass is installed, install it if not (assuming Ubuntu/Debian-based WSL)
if ! command -v sshpass &> /dev/null; then
    echo "sshpass not found. Installing it..."
    sudo apt update && sudo apt install -y sshpass
fi

# Create LOCAL_DIR if it doesn't exist
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Local directory doesn't exist. Creating it..."
    mkdir -p "$LOCAL_DIR"
fi

# Get list of subdirectories from remote server
subdirs=$(sshpass -p "$PASSWORD" ssh "$REMOTE_USER@$REMOTE_HOST" "ls -d $REMOTE_PATH/*/")

# Check if we got any subdirectories
if [ -z "$subdirs" ]; then
    echo "No subdirectories found in remote path $REMOTE_PATH"
    exit 1
fi

# Loop through each subdirectory and sync it
for dir in $subdirs; do
    dir_name=$(basename "$dir")
    echo "Syncing directory: $dir_name"
    
    # Use sshpass with rsync to sync from remote to local
    sshpass -p "$PASSWORD" rsync -rz --info=progress2 --info=name0 -e ssh \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$dir_name/" "$LOCAL_DIR/$dir_name"
    
    if [ $? -eq 0 ]; then
        echo "Successfully synced $dir_name"
    else
        echo "Error syncing $dir_name"
    fi
done

echo "Sync process completed."