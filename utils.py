import os
import re
import openslide
import cv2
import numpy as np

import matplotlib.pyplot as plt

def visualize_photos_on_thumbnail(target_folder):
    path = f"./patches_tiles/{target_folder}/excluded"
    slide_path = f"./{target_folder}.mrxs"
    output_dir = f"./outputs/{target_folder}"
    os.makedirs(output_dir, exist_ok=True)

    items = [item for item in os.listdir(path) if item.endswith(".png") and os.path.isfile(os.path.join(path, item))]
    if not items:
        print(f"No patch images found in {path}")
        return

    slide = openslide.OpenSlide(slide_path)
    thumbnail = slide.get_thumbnail((1200, 1200)).convert("RGB")
    thumbnail_rgb = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)

    slide_w, slide_h = slide.dimensions
    thumb_h, thumb_w = thumbnail_rgb.shape[:2]
    scale_x = thumb_w / float(slide_w)
    scale_y = thumb_h / float(slide_h)

    dot_color = (0, 0, 255)
    dot_radius = 3

    for item in items:
        match = re.match(r"patch_(\d+)_(\d+)\.png$", item)
        if not match:
            continue

        x_px = int(match.group(1))
        y_px = int(match.group(2))

        patch_path = os.path.join(path, item)
        patch = cv2.imread(patch_path)
        if patch is None:
            continue

        patch_h, patch_w = patch.shape[:2]
        x_center = x_px + patch_w // 2
        y_center = y_px + patch_h // 2

        x_thumb = int(round(x_center * scale_x))
        y_thumb = int(round(y_center * scale_y))

        cv2.circle(thumbnail_rgb, (x_thumb, y_thumb), dot_radius, dot_color, -1)

    visualization_path = os.path.join(output_dir, "filtered_thumbnail_dots.png")

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(thumbnail_rgb, cv2.COLOR_BGR2RGB))
    plt.title(f"Filtered patches on thumbnail: {target_folder}")
    plt.axis("off")
    plt.savefig(visualization_path, bbox_inches="tight", dpi=200)
    plt.close()

    slide.close()
    print(f"Thumbnail visualization saved to: {visualization_path}")