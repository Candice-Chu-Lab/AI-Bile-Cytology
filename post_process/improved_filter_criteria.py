import cv2
import os
import numpy as np


def background_ratio(patch, threshold=220):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    white_pixels = (gray > threshold).sum()
    return white_pixels / gray.size


def stained_pixel_ratio(patch):
    """
    Detect blue/purple stained pixels.
    This is useful for bacteria/cell-like objects on bright background.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]  # hue
    s = hsv[:, :, 1]  # saturation
    v = hsv[:, :, 2]  # brightness/value

    # Blue / purple regions usually have:
    # - enough saturation
    # - not too bright
    # - hue in blue/purple range
    stained_mask = (
        (s > 40) &
        (v < 230) &
        (h > 90) & (h < 170)
    )

    return stained_mask.sum() / stained_mask.size


# Calculate the count of connected components in the stained mask, which can indicate presence of bacteria/cells
def stained_component_count(patch, min_area=5):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = (
        (s > 40) &
        (v < 230) &
        (h > 90) & (h < 170)
    ).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    count = 0
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            count += 1

    return count

def is_empty_patch(patch):
    bg_ratio = background_ratio(patch)
    stain_ratio = stained_pixel_ratio(patch)
    component_count = stained_component_count(patch)

    return (
        bg_ratio > 0.90 and
        stain_ratio < 0.0003 and
        component_count == 0
    )





if __name__ == "__main__":
    empty_patch_cnt = 0

    slide_path = "DS_B04R_04S"
    base_path = f"./patches_tiles/{slide_path}"

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Patch folder not found: {base_path}")

    excluded_dir = f"{base_path}/excluded"
    os.makedirs(excluded_dir, exist_ok=True)

    for filename in os.listdir(base_path):
        if filename.endswith(".png"):
            patch_path = os.path.join(base_path, filename)
            patch = cv2.imread(patch_path)

            if patch is None:
                print(f"Skipping unreadable image: {patch_path}")
                continue

            if is_empty_patch(patch):
                empty_patch_cnt += 1
                excluded_path = os.path.join(excluded_dir, filename)
                os.replace(patch_path, excluded_path)

    print(f"Total empty patches: {empty_patch_cnt}")