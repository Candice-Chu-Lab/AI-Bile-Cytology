import os
import re
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import openslide
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import visualize_photos_on_thumbnail


def folder_separation(target_folder, num_photos = 400):
    print("Separating photos into folders...")
    path = f"./patches_tiles/{target_folder}"
    path_section = f"./patches_tiles/{target_folder}/section"
    os.makedirs(path_section, exist_ok=True)
    # exclude any folder
    items = [item for item in os.listdir(path) if os.path.isfile(os.path.join(path, item))]

    # separate the photos into folders, each with the same number of photos (num_photos)
    for i in range(0, len(items), num_photos):
        section_items = items[i:i+num_photos]
        section_folder = f"{path_section}/section_{i//num_photos + 1}"
        os.makedirs(section_folder, exist_ok=True)

        for item in section_items:
            src_path = os.path.join(path, item)
            dst_path = os.path.join(section_folder, item)
            # copy the photo to the new folder
            shutil.copy2(src_path, dst_path)

    print(f"Separated {len(items)} photos into {len(items) // num_photos + 1} sections in {path_section}")






# Can we visualize the photos filtered on a thumbnail image with dots?
# the filename format: patch_53720_124071.png



if __name__ == "__main__":
    target_folder = "DS_B04R_04S"
    # folder_separation(target_folder)
    visualize_photos_on_thumbnail(target_folder)





