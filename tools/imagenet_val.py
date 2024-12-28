import os
import shutil

# Path to the folder where all the images are unzipped
source_folder = '/cluster/project/sachan/callen/data_alice/ILSVRC2012_img/val'

# Destination folder where you want to organize the images by class
destination_folder = '/cluster/project/sachan/callen/data_alice/ILSVRC2012_img/val_fixed'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".JPEG"):
        # Extract class number from the filename (assuming the format "classNumber_imageID.JPEG")
        class_number = filename.split('_')[0]

        # Define the path for the class folder
        class_folder = os.path.join(destination_folder, class_number)

        # Create the class folder if it doesn't exist
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Move the image to the class folder
        src_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(class_folder, filename)
        shutil.move(src_path, dest_path)

print("Images have been organized by class.")