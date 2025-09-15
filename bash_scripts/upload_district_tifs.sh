#!/bin/bash

SOURCE_DIR="/home/ccds/Desktop/Brickfield/geotiff_districts"
DEST="temp@103.159.2.125:/opt/models/Bangladesh_2019/64_dist"
PASSWORD="34512312"  # Replace with your actual password

# Install sshpass if not already installed
# sudo apt-get install sshpass  # On Ubuntu/Debian
# sudo yum install sshpass      # On CentOS/RHEL

for file in "$SOURCE_DIR"/*.tif; do
    if [ -f "$file" ]; then
        echo "Transferring: $(basename "$file")"
        rsync -rz --info=progress2 --info=name0 -e "sshpass -p '$PASSWORD' ssh" "$file" "$DEST"
    else
        echo "No .tif files found in $SOURCE_DIR"
        break
    fi
done
