import os
import shutil

# Original source directory containing your files
source_dir = r"CURRENT_PATH\preprocessing" # Change this to your actual path

# New directory 'youtube' inside preprocessing
preprocessing_dir = os.path.dirname(source_dir)  # gets the 'preprocessing' folder
dest_dir = os.path.join(preprocessing_dir, "youtube")
os.makedirs(dest_dir, exist_ok=True)  # create youtube1 if it doesn't exist

# Iterate over all items in the source directory
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    
    # Only process files (skip folders)
    if os.path.isfile(file_path):
        # Create a new folder with the file name inside youtube1
        folder_path = os.path.join(dest_dir, filename)
        os.makedirs(folder_path, exist_ok=True)
        
        # Move the file into the newly created folder
        shutil.move(file_path, os.path.join(folder_path, filename))

print("Files moved and restructured inside 'youtube'!")
