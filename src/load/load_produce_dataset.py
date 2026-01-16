import kagglehub
import os
import shutil

# Get the project root directory (one level up from src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Download latest version
path = kagglehub.dataset_download("henningheyen/lvis-fruits-and-vegetables-dataset")

print("Path to dataset files:", os.path.abspath(path))

# Define destination path relative to project root
destination = os.path.join(project_root, "assets", "produce_dataset")

# Create the destination directory if it doesn't exist
os.makedirs(os.path.dirname(destination), exist_ok=True)

# Copy the dataset to the destination
shutil.copytree(path, destination, dirs_exist_ok=True)
print(f"Dataset copied to {os.path.abspath(destination)}")

# List the contents of the copied directory
print(f"Contents of {os.path.abspath(destination)}:")
for root, dirs, files in os.walk(destination):
    level = root.replace(destination, "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")

print("Done!")