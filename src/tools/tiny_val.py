import os
import shutil

# Path to your text file
txt_file_path = "/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/val/val_annotations.txt"

# Directory containing the images
image_directory = "/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/val/images"

# Directory where the images will be organized
output_directory = "/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/val_new"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read the text file line by line
with open(txt_file_path, 'r') as file:
    for line in file:
        # Split the line by tab or spaces (depending on the format in your file)
        parts = line.strip().split()
        
        # Extract the image filename and class
        image_filename = parts[0]
        image_class = parts[1]
        
        # Create the class directory if it doesn't exist
        class_directory = os.path.join(output_directory, image_class)
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)
        
        # Source file path (current location of the image)
        source_file = os.path.join(image_directory, image_filename)
        
        # Destination file path (new location in the class directory)
        destination_file = os.path.join(class_directory, "images", image_filename)
        
        # Move the file
        if os.path.exists(source_file):
            shutil.move(source_file, destination_file)
        else:
            print(f"Warning: {source_file} does not exist and cannot be moved.")

print("Files have been organized successfully.")
