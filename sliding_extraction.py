import openslide
import numpy as np
from obtain_ring import detect_inner_ring_location
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
from pathlib import Path

from dataclasses import dataclass

from post_process.improved_filter_criteria import should_exclude_patch



@dataclass
class Args:
    slide_path: str
    patch_size: int = 1024
    max_saved_patches: int = 1500
    run_filtering: bool = False
    separate_folder: bool = False
    select_patch: bool = False
    save_dir: str = r"D:\bile_sample\result"
    # output_dir: str = "./patches_tiles/outputs"
    folder_num_images: int = 400 # number of images per folder when separate_folder is True



def select_region_on_thumbnail(thumbnail_rgb):
    """
    Let the user draw a circular region on the thumbnail. The function
    returns the bounding box of the circle as (x1, y1, x2, y2) in
    thumbnail pixel coordinates.

    Controls:
    1. Drag with left mouse button to select an ellipse; it will be
       converted to a circle (largest dimension used as diameter).
    2. Press Enter to confirm.
    3. Close the window if needed.
    """
    selected = {}

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(thumbnail_rgb)
    ax.set_title("Drag to select circular region, then press Enter")
    ax.axis("off")

    def onselect(eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None:
            return
        if erelease.xdata is None or erelease.ydata is None:
            return

        x1, y1 = float(eclick.xdata), float(eclick.ydata)
        x2, y2 = float(erelease.xdata), float(erelease.ydata)

        # Use the diagonal the user dragged to create a circle: take the
        # larger of width/height as the diameter so the selection becomes
        # a circle (centered on the drag midpoint).
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        radius = int(round(max(width, height) / 2.0))

        selected["x1"] = cx - radius
        selected["y1"] = cy - radius
        selected["x2"] = cx + radius
        selected["y2"] = cy + radius

        print(
            f"Selected circular thumbnail bounding box: "
            f"({selected['x1']}, {selected['y1']}) "
            f"to ({selected['x2']}, {selected['y2']})"
        )

    def on_key(event):
        if event.key == "enter":
            if selected:
                plt.close(fig)
            else:
                print("Please drag a circular region before pressing Enter.")

    # Important: keep selector alive by assigning it to a variable
    selector = EllipseSelector(
        ax,
        onselect,
        useblit=False,
        button=1,
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


def patch_overlap_fraction(inner_mask, x, y, patch_w, patch_h):
    """
    Return fraction of pixels inside the mask for the given thumbnail patch.
    inner_mask is expected to be non-zero for inside region (e.g., 255).
    """
    patch = inner_mask[y:y + patch_h, x:x + patch_w]

    if patch.size == 0:
        return 0.0

    white = np.count_nonzero(patch)
    area = patch.shape[0] * patch.shape[1]

    return float(white) / float(area)


def save_patch(slide, x, y, patch_size, save_dir, patch_name):
    patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")

    filename = os.path.join(save_dir, f"{patch_name}_patch_{x}_{y}.png")
    patch.save(filename)



def run_extraction(args):

    slide_path = args.slide_path
    base_name = os.path.basename(slide_path).split(".")[0]

    patch_size = args.patch_size
    max_saved_patches = args.max_saved_patches

    # create save directory and output directory for this slide
    save_dir = f"{args.save_dir}/{base_name}"
    output_dir = f"{args.save_dir}/{base_name}/output"
    # output_dir = f"{args.output_dir}/{base_name}"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    slide = openslide.OpenSlide(slide_path)
    W, H = slide.dimensions

    print(f"Slide dimensions: {W} x {H}")

    ring_info = None
    try:
        ring_info = detect_inner_ring_location(slide_path, show_debug=False)
    except Exception as e:
        print(f"Warning: detect_inner_ring_location failed: {e}")
        ring_info = None

    if ring_info:
        inner_mask = ring_info.get("inner_mask_thumbnail")
        thumbnail_rgb = ring_info.get("thumbnail_rgb")
        thumbnail_shape = ring_info.get("thumbnail_shape")
        slide_shape = ring_info.get("slide_shape")
        used_fallback_mask = False
    else:
        inner_mask = None
        # Fallback: build a thumbnail directly from the slide so the
        # rest of the pipeline can continue even if ring detection fails.
        max_dim = 1024
        scale = max_dim / float(max(W, H))
        thumb_w = max(1, int(round(W * scale)))
        thumb_h = max(1, int(round(H * scale)))
        try:
            thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
            thumbnail_rgb = np.array(thumb)[:, :, ::-1]
        except Exception:
            # Last-resort: downsample level-0 region
            region = slide.read_region((0, 0), slide.level_count - 1, (thumb_w, thumb_h)).convert("RGB")
            thumbnail_rgb = np.array(region)[:, :, ::-1]

        thumbnail_shape = thumbnail_rgb.shape[:2]
        slide_shape = (H, W)
        used_fallback_mask = True

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
    # save the copy for overlay image to a specific path 
    overlay_path = os.path.join(output_dir, "thumbnail_overlay_initial.png")
    cv2.imwrite(overlay_path, overlay)

    # ------------------------------------------------------------
    # Step 1: User selects circular region on thumbnail
    # ------------------------------------------------------------
    x1_thumb, y1_thumb, x2_thumb, y2_thumb = select_region_on_thumbnail(thumbnail_rgb)

    # ------------------------------------------------------------
    # Step 2: Convert thumbnail selection bounding box to level-0 slide rectangle
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

    print(f"Selected slide-level extraction region of {base_name}:")
    print(f"x: {x1_slide} to {x2_slide}")
    print(f"y: {y1_slide} to {y2_slide}")

    # log to a text file
    with open(os.path.join(output_dir, f"extraction_log_{base_name}.txt"), "w") as log_file:
        log_file.write(f"Slide dimensions: {W} x {H}\n")
        log_file.write(f"Slide bounds:\n")
        log_file.write(f"x: {bounds_x} to {x_end}\n")
        log_file.write(f"y: {bounds_y} to {y_end}\n")
        log_file.write("Selected slide-level extraction region:\n")
        log_file.write(f"x: {x1_slide} to {x2_slide}\n")
        log_file.write(f"y: {y1_slide} to {y2_slide}\n")
        # Log circular selection info (thumbnail coords and mapped to slide)
        cx_thumb_log = int(round((x1_thumb + x2_thumb) / 2.0))
        cy_thumb_log = int(round((y1_thumb + y2_thumb) / 2.0))
        r_thumb_log = int(round(max((x2_thumb - x1_thumb), (y2_thumb - y1_thumb)) / 2.0))

        thumb_h_log, thumb_w_log = thumbnail_shape
        slide_h_log, slide_w_log = slide_shape
        scale_x_log = slide_w_log / float(thumb_w_log)
        scale_y_log = slide_h_log / float(thumb_h_log)

        cx_slide_log = int(round(cx_thumb_log * scale_x_log))
        cy_slide_log = int(round(cy_thumb_log * scale_y_log))
        r_slide_log = int(round(r_thumb_log * ((scale_x_log + scale_y_log) / 2.0)))

        log_file.write("User circular selection (thumbnail px):\n")
        log_file.write(f"center: ({cx_thumb_log}, {cy_thumb_log}), radius: {r_thumb_log}\n")
        log_file.write("User circular selection (approx slide px):\n")
        log_file.write(f"center: ({cx_slide_log}, {cy_slide_log}), radius: {r_slide_log}\n")

    if x2_slide <= x1_slide or y2_slide <= y1_slide:
        raise RuntimeError("Selected region is invalid after clamping to slide bounds.")

    # Draw circular overlay corresponding to the selected bounding box
    cx_thumb = int(round((x1_thumb + x2_thumb) / 2.0))
    cy_thumb = int(round((y1_thumb + y2_thumb) / 2.0))
    r_thumb = int(round(max((x2_thumb - x1_thumb), (y2_thumb - y1_thumb)) / 2.0))
    cv2.circle(overlay, (cx_thumb, cy_thumb), r_thumb, (0, 255, 0), 3)

    # Use the user-drawn circular mask for extraction regardless of
    # whether automatic ring detection succeeded.
    print("Using user-drawn circular mask for extraction (overrides detection).")
    h_thumb, w_thumb = thumbnail_rgb.shape[:2]
    inner_mask = np.zeros((h_thumb, w_thumb), dtype=np.uint8)
    cv2.circle(inner_mask, (cx_thumb, cy_thumb), r_thumb, 255, -1)
    used_fallback_mask = True

    saved_count = 0

    print("Running patch extraction...")
    print("Note: This process may take several minutes depending on the number of patches.")
    print("Scanning selected region for valid patches...")

    # ------------------------------------------------------------
    # Step 4: Extract patches only inside selected bounding box
    # ------------------------------------------------------------

    for y_px in range(y1_slide, y2_slide - patch_size + 1, patch_size):
        for x_px in range(x1_slide, x2_slide - patch_size + 1, patch_size):
            x_thumb, y_thumb, pw_t, ph_t = slide_patch_to_thumbnail_patch(
                x_px, y_px, patch_size, slide_shape, thumbnail_shape
            )

            if used_fallback_mask:
                # Require a high overlap fraction with the circular mask
                # (avoids saving patches that only slightly overlap the circle).
                frac = patch_overlap_fraction(inner_mask, x_thumb, y_thumb, pw_t, ph_t)
                inside_status = frac >= 0.9
            else:
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
    #plt.show()

    print(f"Overlay saved to: {overlay_path}")

    slide.close()


def run_filter(args):
    from post_process import background_info, stained_pixel_ratio, stained_component_count, is_empty_patch
    print("Running post-extraction filtering...")
    empty_patch_cnt = 0

    slide_path = Path(args.slide_path).stem
    base_path = f"{args.save_dir}/{slide_path}"

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
        
        if should_exclude_patch(patch):
            empty_patch_cnt += 1
            excluded_path = os.path.join(excluded_dir, filename)
            os.replace(patch_path, excluded_path)

    print(f"Total empty patches: {empty_patch_cnt}")


# example: python sliding_extraction.py --slide_path "D:\bile_sample\DS_B04R_07S.mrxs" --run_filtering --select_patch
if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)
    run_extraction(args)
    # filter patches
    if args.run_filtering:
        run_filter(args)
    
    if args.select_patch:
        from post_process.select_patches import draw_thumbnail, select_patch
        slide_name = Path(args.slide_path).stem.split(".")[0]
        selected_patches = select_patch(slide_name=slide_name)
        draw_thumbnail(slide_name, selected_patches)