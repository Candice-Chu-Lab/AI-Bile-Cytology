'''
This program select pataches from the original image and save them in the "selected" folder
There will be a thumbnail image that plots what patches are selected
'''

import os
import cv2
import numpy as np


base_path= r"D:\bile_sample\result"

def select_patch(slide_name):
    slide_path = os.path.join(base_path, slide_name)
    save_dir = os.path.join(slide_path, 'selected')
    os.makedirs(save_dir, exist_ok=True)
    total_patches = [i for i in os.listdir(slide_path) if i.startswith('DS_')]
    random_patches = np.random.choice(total_patches, size=450, replace=False)
    for patch in random_patches:
        patch_path = os.path.join(slide_path, patch)
        # save patches to the "selected" folder
        save_path = os.path.join(save_dir, patch)
        # copy the patch to the "selected" folder
        cv2.imwrite(save_path, cv2.imread(patch_path))


    return random_patches

# draw the thumbnail image
def draw_thumbnail(slide_name, selected_patches):
    # the format of the patch name is DS_A09R_05S_patch_55469_124796.png
    import re

    slide_path = os.path.join(base_path, slide_name)


    thumb_path = os.path.join(slide_path, "output", 'thumbnail_overlay_initial.png')
    thumb = cv2.imread(thumb_path)
    if thumb is None:
        return None

    h_thumb, w_thumb = thumb.shape[:2]

    # parse coordinates from patch filenames
    coords = []
    for name in selected_patches:
        # accept either full path or basename
        bn = os.path.basename(name)
        m = re.search(r'patch_(\d+)_(\d+)', bn)
        if m:
            x = int(m.group(1))
            y = int(m.group(2))
            coords.append((x, y))
        else:
            # try last two numeric groups before extension
            parts = os.path.splitext(bn)[0].split('_')
            if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
                x = int(parts[-2])
                y = int(parts[-1])
                coords.append((x, y))

    if not coords:
        return thumb

    # assume patch size (pixels) — adjust if your patches use a different size
    PATCH_SIZE = 1024

    xs = [x for x, y in coords]
    ys = [y for x, y in coords]
    max_x = max(xs) + PATCH_SIZE
    max_y = max(ys) + PATCH_SIZE

    # compute scale from full-slide pixel coords -> thumbnail pixels
    # Prefer using recorded slide dimensions from the extraction log so the
    # mapping is accurate. Fallback to the previous heuristic when the log
    # is missing.
    slide_w = None
    slide_h = None
    try:
        log_path = os.path.join(slide_path, "output", f"extraction_log_{slide_name}.txt")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Slide dimensions:"):
                        # expected format: "Slide dimensions: {W} x {H}"
                        parts = line.split(":", 1)[1].strip()
                        if "x" in parts:
                            w_s, h_s = parts.split("x")
                            slide_w = int(w_s)
                            slide_h = int(h_s)
                        break
    except Exception:
        slide_w = slide_h = None

    if slide_w and slide_h:
        scale_x = w_thumb / float(slide_w) if slide_w > 0 else 1.0
        scale_y = h_thumb / float(slide_h) if slide_h > 0 else 1.0
    else:
        # fallback: use max coordinates derived from patches (legacy behaviour)
        scale_x = w_thumb / float(max_x) if max_x > 0 else 1.0
        scale_y = h_thumb / float(max_y) if max_y > 0 else 1.0

    # draw rectangles for each selected patch
    out = thumb.copy()
    for (x, y) in coords:
        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + PATCH_SIZE) * scale_x))
        y2 = int(round((y + PATCH_SIZE) * scale_y))
        # clamp to thumbnail bounds
        x1 = max(0, min(w_thumb - 1, x1))
        y1 = max(0, min(h_thumb - 1, y1))
        x2 = max(0, min(w_thumb, x2))
        y2 = max(0, min(h_thumb, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # annotate slide name
    cv2.putText(out, slide_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # save result next to thumbnail
    save_dir = os.path.join(slide_path, "output")
    os.makedirs(save_dir, exist_ok=True)
    out_name = f"{slide_name}_thumbnail_selected.png"
    out_path = os.path.join(save_dir, out_name)
    cv2.imwrite(out_path, out)
    return out_path



if __name__ == "__main__":
    slide_name = "DS_A09R_16S"
    selected_patches = select_patch(slide_name)
    thumbnail_path = draw_thumbnail(slide_name, selected_patches)
    # print(f"Selected patches saved. Thumbnail with selections saved at: {thumbnail_path}")