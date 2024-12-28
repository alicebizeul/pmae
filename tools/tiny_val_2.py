import os
import shutil

# Path to the root directory
root_directory = "/cluster/project/sachan/callen/data_alice/tiny-imagenet-200/val"

# Traverse the root directory
for subdir in os.listdir(root_directory):
    subdir_path = os.path.join(root_directory, subdir)

    # Check if it's a directory
    if os.path.isdir(subdir_path):
        # Create the "images" subdirectory within the current subdirectory
        images_subdir = os.path.join(subdir_path, "images")
        if not os.path.exists(images_subdir):
            os.makedirs(images_subdir)

        # Move all files in the current subdirectory to the "images" subdirectory
        for item in os.listdir(subdir_path):
            item_path = os.path.join(subdir_path, item)

            # Move the item if it's a file (not a directory)
            if os.path.isfile(item_path):
                shutil.move(item_path, images_subdir)

print("Directory structure has been updated successfully.")
