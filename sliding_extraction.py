import openslide
import numpy as np
from obtain_ring import detect_inner_ring_location
import os
import cv2
import matplotlib.pyplot as plt

x_um = 7649
y_um = 21572

def um_to_pixels(slide, x_um, y_um, use_bounds_offset=True):
    """
    Convert microns to level-0 slide pixels.

    Set use_bounds_offset=True only if x_um, y_um are measured
    relative to the bounded tissue origin instead of the slide origin.
    """
    mpp_x = float(slide.properties["openslide.mpp-x"])
    mpp_y = float(slide.properties["openslide.mpp-y"])

    x_pixels = int(round(x_um / mpp_x))
    y_pixels = int(round(y_um / mpp_y))

    if use_bounds_offset:
        bounds_x = int(slide.properties.get("openslide.bounds-x", 0))
        bounds_y = int(slide.properties.get("openslide.bounds-y", 0))
        x_pixels += bounds_x
        y_pixels += bounds_y

    return x_pixels, y_pixels


def slide_patch_to_thumbnail_patch(
    x_px,
    y_px,
    patch_size_slide,
    slide_shape,
    thumbnail_shape,
):
    """
    Map a level-0 slide patch to thumbnail coordinates.
    """
    slide_h, slide_w = slide_shape
    thumb_h, thumb_w = thumbnail_shape

    scale_x = thumb_w / float(slide_w)
    scale_y = thumb_h / float(slide_h)

    x_thumb = int(round(x_px * scale_x))
    y_thumb = int(round(y_px * scale_y))
    patch_w_thumb = max(1, int(round(patch_size_slide * scale_x)))
    patch_h_thumb = max(1, int(round(patch_size_slide * scale_y)))

    return x_thumb, y_thumb, patch_w_thumb, patch_h_thumb


def patch_inner(inner_mask, x, y, patch_w, patch_h):
    """
    A patch is entirely within the inner hole if all its pixels are white.
    """
    patch = inner_mask[y:y + patch_h, x:x + patch_w]

    if patch.size == 0:
        return False

    white = np.count_nonzero(patch)
    area = patch.shape[0] * patch.shape[1]

    return white == area


def patch_fully_inside_inner_hole(inner_mask, x, y, patch_w, patch_h):
    patch = inner_mask[y:y + patch_h, x:x + patch_w]
    if patch.size == 0:
        return False
    white = np.count_nonzero(patch)
    area = patch.shape[0] * patch.shape[1]
    return white == area

def save_patch(slide, x, y, patch_size, save_dir):
    patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")

    filename = os.path.join(save_dir, f"patch_{x}_{y}.png")
    patch.save(filename)

def patch_fully_outside_inner_hole(inner_mask, x, y, patch_w, patch_h):
    patch = inner_mask[y:y + patch_h, x:x + patch_w]
    if patch.size == 0:
        return True
    white = np.count_nonzero(patch)
    return white == 0

if __name__ == "__main__":
    slide_path = "DS_B04R_04S.mrxs"
    base_name = os.path.basename(slide_path).split('.')[0]
    patch_size = 512
    save_dir = f"./patches_tiles/{base_name}"

    os.makedirs(save_dir, exist_ok=True)
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.dimensions

    print(f"Slide dimensions: {W} x {H}")

    # detect inner mask + scaling info
    ring_info = detect_inner_ring_location(slide_path, show_debug=True)

    inner_mask = ring_info["inner_mask_thumbnail"]
    thumbnail_shape = ring_info["thumbnail_shape"]
    slide_shape = ring_info["slide_shape"]

    bounds_x = int(slide.properties.get("openslide.bounds-x", 0))
    bounds_y = int(slide.properties.get("openslide.bounds-y", 0))
    bounds_w = int(slide.properties.get("openslide.bounds-width", W - bounds_x))
    bounds_h = int(slide.properties.get("openslide.bounds-height", H - bounds_y))

    x_end = bounds_x + bounds_w
    y_end = bounds_y + bounds_h

    saved_count = 0

    print("Scanning patches...")

    # start from the particular bound

    x_px, y_px = um_to_pixels(slide, x_um, y_um, use_bounds_offset=True)
    overlay = ring_info["thumbnail_rgb"].copy()
    # for y_px in range(y_px, y_end - patch_size + 1, patch_size):
    for x_px in range(x_px, x_end - patch_size + 1, patch_size):

        # map to thumbnail
        x_thumb, y_thumb, pw_t, ph_t = slide_patch_to_thumbnail_patch(
            x_px,
            y_px,
            patch_size,
            slide_shape,
            thumbnail_shape
        )

        # check boundary overlap
        inside_status = patch_inner(
            inner_mask,
            x_thumb,
            y_thumb,
            pw_t,
            ph_t
        )
        

        if inside_status:
            print(f"Saving patch at ({x_px}, {y_px})")
            save_patch(slide, x_px, y_px, patch_size, save_dir)
            saved_count += 1

            # red rectangle on thumbnail
            cv2.rectangle(
                overlay,
                (x_thumb, y_thumb),
                (x_thumb + pw_t, y_thumb + ph_t),
                (255, 0, 0),
                2
            )

            # red dot at top-left
            cv2.circle(overlay, (x_thumb, y_thumb), 4, (255, 0, 0), -1)
        else:
            print(f"Skipping patch at ({x_px}, {y_px}) - not fully inside inner hole")

        if saved_count >= 100:
            break

    print(f"Done. Saved {saved_count} patches.")

    plt.figure(figsize=(8, 12))
    plt.imshow(overlay)
    plt.title("Saved patches on thumbnail")
    plt.axis("off")
    plt.savefig(f"outputs/{base_name}/thumbnail_overlay.png")



    # 7649, 21572