# How many tiles are secluded based on the following criteria:
# patches_files/



import cv2
import os
# unsure: https://chatgpt.com/s/t_69ebd9ce9c5c8191ba031a34a9933e8c

def background_ratio(patch, threshold=200):
    # grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    white_pixels = (gray > threshold).sum()
    return white_pixels / gray.size


if __name__ == "__main__":
    empty_patch_cnt = 0
    # check a specific patches tile
    slide_path = "DS_B04R_04S"
    base_path = f"./patches_tiles/{slide_path}"
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Patch folder not found: {base_path}")

    for filename in os.listdir(base_path):
        if filename.endswith(".png"):
            patch_path = os.path.join(base_path, filename)
            patch = cv2.imread(patch_path) # returns in BGR instead of RGB

            if patch is None:
                print(f"Skipping unreadable image: {patch_path}")
                continue

            if background_ratio(patch) > 0.9:
                # empty patch += 1
                empty_patch_cnt += 1

                # save to the excluded folder
                excluded_dir = f"./patches_excluded/{slide_path}"
                os.makedirs(excluded_dir, exist_ok=True)
                excluded_path = os.path.join(excluded_dir, filename)
                cv2.imwrite(excluded_path, patch)



    print(f"Total empty patches: {empty_patch_cnt}")