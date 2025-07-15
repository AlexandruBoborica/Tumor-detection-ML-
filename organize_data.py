import os
import shutil

# Directories
directory = "./isles"
images_directory = os.path.join(directory, "images")
labels_directory = os.path.join(directory, "labels")
directories_to_create = ["train", "val", "test"]

# Create necessary directories for train, val, test in both images and labels
for dir in directories_to_create:
    os.makedirs(os.path.join(images_directory, dir), exist_ok=True)
    os.makedirs(os.path.join(labels_directory, dir), exist_ok=True)

# Function to move the files based on the .txt file
def move_files(dir_name):
    file = os.path.join(directory, f"{dir_name}.txt")
    with open(file, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        image_path = line.strip()
        image_filename = os.path.basename(image_path)
        label_filename = image_filename.replace(".png", ".txt")

        # Define source and destination paths for image and label
        src_img = os.path.join(images_directory, image_filename)
        dst_img = os.path.join(images_directory, dir_name, image_filename)

        src_lbl = os.path.join(labels_directory, label_filename)
        dst_lbl = os.path.join(labels_directory, dir_name, label_filename)

        # Check if the image and label files exist and move them
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        else:
            print(f"Image not found: {src_img}")

        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        else:
            print(f"Label not found: {src_lbl}")

# Move the files for train, val, and test based on the .txt files
for dir in directories_to_create:
    move_files(dir)

print("All files have been organized into train/val/test folders!")
