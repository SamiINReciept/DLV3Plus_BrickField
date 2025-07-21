import os

# Get the current working directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Construct the path to a file one level up
# Replace 'example.txt' with the file you want to access
file_path = os.path.join(current_dir, "..", "Districts_patched.zip")

# Normalize the path to resolve '..' correctly
file_path = os.path.abspath(file_path)
print(f"File path: {file_path}")

# Check if the file exists
if os.path.exists(file_path):
    print("File exists! You can access it.")
else:
    print("File does not exist or cannot be accessed.")