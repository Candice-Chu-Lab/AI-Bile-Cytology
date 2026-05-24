import openslide
import numpy as np
from obtain_ring import detect_inner_ring_location
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from pathlib import Path

from dataclasses import dataclass


@dataclass
class Args:
    slide_path: str
    patch_size: int = 512
    max_saved_patches: int = 600
    run_filtering: bool = False



def select_region_on_thumbnail(thumbnail_rgb):
    """
    Let the user draw a rectangle on the thumbnail.

    Controls:
    1. Drag with left mouse button to select a rectangle.
    2. Press Enter to confirm.
    3. Close the window if needed.

    Returns:
        x1, y1, x2, y2 in thumbnail pixel coordinates.
    """
    selected = {}

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(thumbnail_rgb)
    ax.set_title("Drag to select region, then press Enter")
    ax.axis("off")

    def onselect(eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None:
            return
        if erelease.xdata is None or erelease.ydata is None:
            return

        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        selected["x1"] = min(x1, x2)
        selected["y1"] = min(y1, y2)
        selected["x2"] = max(x1, x2)
        selected["y2"] = max(y1, y2)

        print(
            f"Selected thumbnail region: "
            f"({selected['x1']}, {selected['y1']}) "
            f"to ({selected['x2']}, {selected['y2']})"
        )

    def on_key(event):
        if event.key == "enter":
            if selected:
                plt.close(fig)
            else:
                print("Please drag a rectangle before pressing Enter.")

    # Important: keep selector alive by assigning it to a variable
    selector = RectangleSelector(
        ax,
        onselect,
        useblit=False,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    # Also keep it referenced until after plt.show()
    _ = selector

    if not selected:
        raise RuntimeError("No region selected.")

    return selected["x1"], selected["y1"], selected["x2"], selected["y2"]

# x_um = 7649
# y_um = 21572

# def um_to_pixels(slide, x_um, y_um, use_bounds_offset=True):
#     """
#     Convert microns to level-0 slide pixels.

#     Set use_bounds_offset=True only if x_um, y_um are measured
#     relative to the bounded tissue origin instead of the slide origin.
#     """
#     mpp_x = float(slide.properties["openslide.mpp-x"])
#     mpp_y = float(slide.properties["openslide.mpp-y"])

#     x_pixels = int(round(x_um / mpp_x))
#     y_pixels = int(round(y_um / mpp_y))

#     if use_bounds_offset:
#         bounds_x = int(slide.properties.get("openslide.bounds-x", 0))
#         bounds_y = int(slide.properties.get("openslide.bounds-y", 0))
#         x_pixels += bounds_x
#         y_pixels += bounds_y

#     return x_pixels, y_pixels


# def slide_patch_to_thumbnail_patch(
#     x_px,
#     y_px,
#     patch_size_slide,
#     slide_shape,
#     thumbnail_shape,
# ):
#     """
#     Map a level-0 slide patch to thumbnail coordinates.
#     """
#     slide_h, slide_w = slide_shape
#     thumb_h, thumb_w = thumbnail_shape

#     scale_x = thumb_w / float(slide_w)
#     scale_y = thumb_h / float(slide_h)

#     x_thumb = int(round(x_px * scale_x))
#     y_thumb = int(round(y_px * scale_y))
#     patch_w_thumb = max(1, int(round(patch_size_slide * scale_x)))
#     patch_h_thumb = max(1, int(round(patch_size_slide * scale_y)))

#     return x_thumb, y_thumb, patch_w_thumb, patch_h_thumb



def thumbnail_rect_to_slide_rect(
    x1_thumb,
    y1_thumb,
    x2_thumb,
    y2_thumb,
    slide_shape,
    thumbnail_shape,
):
    """
    Convert a rectangle from thumbnail coordinates to level-0 slide coordinates.

    Args:
        slide_shape: (slide_h, slide_w)
        thumbnail_shape: (thumb_h, thumb_w)

    Returns:
        x1_slide, y1_slide, x2_slide, y2_slide
    """
    slide_h, slide_w = slide_shape
    thumb_h, thumb_w = thumbnail_shape

    scale_x = slide_w / float(thumb_w)
    scale_y = slide_h / float(thumb_h)

    x1_slide = int(round(x1_thumb * scale_x))
    y1_slide = int(round(y1_thumb * scale_y))
    x2_slide = int(round(x2_thumb * scale_x))
    y2_slide = int(round(y2_thumb * scale_y))

    return x1_slide, y1_slide, x2_slide, y2_slide

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


def save_patch(slide, x, y, patch_size, save_dir, patch_name):
    patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")

    filename = os.path.join(save_dir, f"{patch_name}_patch_{x}_{y}.png")
    patch.save(filename)



def run_extraction(args):
    slide_path = args.slide_path
    base_name = os.path.basename(slide_path).split(".")[0]

    patch_size = args.patch_size
    max_saved_patches = args.max_saved_patches

    save_dir = f"./patches_tiles/{base_name}"
    output_dir = f"./patches_tiles/outputs/{base_name}"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    slide = openslide.OpenSlide(slide_path)
    W, H = slide.dimensions

    print(f"Slide dimensions: {W} x {H}")

    ring_info = detect_inner_ring_location(slide_path, show_debug=False)

    inner_mask = ring_info["inner_mask_thumbnail"]
    thumbnail_rgb = ring_info["thumbnail_rgb"]
    thumbnail_shape = ring_info["thumbnail_shape"]
    slide_shape = ring_info["slide_shape"]

    bounds_x = int(slide.properties.get("openslide.bounds-x", 0))
    bounds_y = int(slide.properties.get("openslide.bounds-y", 0))
    bounds_w = int(slide.properties.get("openslide.bounds-width", W - bounds_x))
    bounds_h = int(slide.properties.get("openslide.bounds-height", H - bounds_y))

    x_end = bounds_x + bounds_w
    y_end = bounds_y + bounds_h

    print(f"Slide bounds:")
    print(f"x: {bounds_x} to {x_end}")
    print(f"y: {bounds_y} to {y_end}")

    # Make overlay image for visualization
    overlay = thumbnail_rgb.copy()

    # ------------------------------------------------------------
    # Step 1: User selects rectangle on thumbnail
    # ------------------------------------------------------------
    x1_thumb, y1_thumb, x2_thumb, y2_thumb = select_region_on_thumbnail(thumbnail_rgb)

    # ------------------------------------------------------------
    # Step 2: Convert thumbnail rectangle to level-0 slide rectangle
    # ------------------------------------------------------------
    x1_slide, y1_slide, x2_slide, y2_slide = thumbnail_rect_to_slide_rect(
        x1_thumb, y1_thumb, x2_thumb, y2_thumb, slide_shape, thumbnail_shape
    )

    # ------------------------------------------------------------
    # Step 3: Clamp selected region to valid tissue bounds
    # ------------------------------------------------------------
    x1_slide = max(bounds_x, x1_slide)
    y1_slide = max(bounds_y, y1_slide)
    x2_slide = min(x_end, x2_slide)
    y2_slide = min(y_end, y2_slide)

    print("Selected slide-level extraction region:")
    print(f"x: {x1_slide} to {x2_slide}")
    print(f"y: {y1_slide} to {y2_slide}")

    if x2_slide <= x1_slide or y2_slide <= y1_slide:
        raise RuntimeError("Selected region is invalid after clamping to slide bounds.")

    cv2.rectangle(overlay, (x1_thumb, y1_thumb), (x2_thumb, y2_thumb), (0, 255, 0), 3)

    saved_count = 0

    print("Scanning selected region for valid patches...")

    # ------------------------------------------------------------
    # Step 4: Extract patches only inside selected rectangle
    # ------------------------------------------------------------

    for y_px in range(y1_slide, y2_slide - patch_size + 1, patch_size):
        for x_px in range(x1_slide, x2_slide - patch_size + 1, patch_size):
            x_thumb, y_thumb, pw_t, ph_t = slide_patch_to_thumbnail_patch(
                x_px, y_px, patch_size, slide_shape, thumbnail_shape
            )

            inside_status = patch_inner(inner_mask, x_thumb, y_thumb, pw_t, ph_t)

            if inside_status:
                print(f"Saving patch at ({x_px}, {y_px})")
                save_patch(slide, x_px, y_px, patch_size, save_dir, Path(args.slide_path).stem)
                saved_count += 1

                cv2.rectangle(
                    overlay, (x_thumb, y_thumb), (x_thumb + pw_t, y_thumb + ph_t), (255, 0, 0), 2
                )
                cv2.circle(overlay, (x_thumb, y_thumb), 4, (255, 0, 0), -1)

            else:
                print(f"Skipping patch at ({x_px}, {y_px}) - not fully inside inner hole")

            if saved_count >= max_saved_patches:
                break

        if saved_count >= max_saved_patches:
            break

    print(f"Done. Saved {saved_count} patches.")

    # ------------------------------------------------------------
    # Step 5: Save thumbnail overlay visualization
    # ------------------------------------------------------------

    overlay_path = os.path.join(output_dir, "thumbnail_overlay.png")

    plt.figure(figsize=(8, 12))
    plt.imshow(overlay)
    plt.title("Selected region and saved patches on thumbnail")
    plt.axis("off")
    plt.savefig(overlay_path, bbox_inches="tight", dpi=200)
    plt.show()

    print(f"Overlay saved to: {overlay_path}")

    slide.close()


def run_filter(args):
    from post_process import background_ratio, stained_pixel_ratio, stained_component_count, is_empty_patch
    print("Running post-extraction filtering...")
    empty_patch_cnt = 0

    slide_path = Path(args.slide_path).stem
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

if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)
    run_extraction(args)
    if args.run_filtering:
        run_filter(args)