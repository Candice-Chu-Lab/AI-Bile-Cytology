import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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




def blur_score(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Larger value = sharper image
    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return score

def is_blurry_patch(patch, blur_threshold=50):
    score = blur_score(patch)
    return score < blur_threshold

def is_empty_patch(patch):
    info = get_patch_filter_info(patch)
    return info["is_empty"]


def stained_component_count(
    patch,
    min_area=30,
    max_area=500,
    min_aspect_ratio=1.5,
    s_threshold=35,
    v_threshold=220,
    h_low=90,
    h_high=175,
    use_morphology=False
):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = (
        (s > s_threshold) &
        (v < v_threshold) &
        (h > h_low) &
        (h < h_high)
    ).astype(np.uint8)

    if use_morphology:
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    refined_stain_ratio = mask.sum() / mask.size

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    kept_boxes = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area or area > max_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]

        aspect_ratio = max(w, h_box) / max(1, min(w, h_box))

        if aspect_ratio < min_aspect_ratio:
            continue

        kept_boxes.append((x, y, w, h_box, area, aspect_ratio))

    count = len(kept_boxes)

    return count, kept_boxes, mask, refined_stain_ratio



def get_patch_filter_info(patch):
    """
    Compute all measurements needed for filtering and visualization.
    """

    bg_ratio = background_ratio(patch)
    broad_stain_ratio = stained_pixel_ratio(patch)

    component_count, kept_boxes, refined_mask, refined_stain_ratio = stained_component_count(
        patch
    )

    is_empty = (
        bg_ratio > 0.90 and
        refined_stain_ratio < 0.0003 and
        component_count == 0
    )

    return {
        "is_empty": is_empty,
        "bg_ratio": bg_ratio,
        "broad_stain_ratio": broad_stain_ratio,
        "refined_stain_ratio": refined_stain_ratio,
        "component_count": component_count,
        "kept_boxes": kept_boxes,
        "refined_mask": refined_mask,
    }

def visualize_patch_filter(patch):
    info = get_patch_filter_info(patch)

    is_empty = info["is_empty"]
    bg_ratio = info["bg_ratio"]
    broad_stain_ratio = info["broad_stain_ratio"]
    refined_stain_ratio = info["refined_stain_ratio"]
    component_count = info["component_count"]
    kept_boxes = info["kept_boxes"]
    refined_mask = info["refined_mask"]

    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    background_mask = gray > 220

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    broad_stained_mask = (
        (s > 40) &
        (v < 230) &
        (h > 90) & (h < 170)
    )

    kept_vis = patch_rgb.copy()

    for box in kept_boxes:
        x, y, w, h_box, area, aspect_ratio = box
        cv2.rectangle(
            kept_vis,
            (x, y),
            (x + w, y + h_box),
            (255, 0, 0),
            1
        )

    plt.figure(figsize=(18, 10))

    plt.subplot(2, 4, 1)
    plt.imshow(patch_rgb)
    plt.title("1. Original patch")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("2. Grayscale")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(background_mask, cmap="gray")
    plt.title(f"3. Bright background mask\nbg_ratio={bg_ratio:.4f}")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(h, cmap="hsv")
    plt.title("4. Hue channel")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(broad_stained_mask, cmap="gray")
    plt.title(f"5. Broad stained mask\nratio={broad_stain_ratio:.6f}")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(refined_mask, cmap="gray")
    plt.title(f"6. Refined mask\nratio={refined_stain_ratio:.6f}")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(kept_vis)
    plt.title(f"7. Kept components\ncount={component_count}")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    decision_text = (
        f"is_empty_patch = {is_empty}\n\n"
        f"bg_ratio = {bg_ratio:.6f}\n"
        f"broad_stain_ratio = {broad_stain_ratio:.6f}\n"
        f"refined_stain_ratio = {refined_stain_ratio:.6f}\n"
        f"component_count = {component_count}"
    )
    plt.text(0.05, 0.5, decision_text, fontsize=14)
    plt.axis("off")

    plt.suptitle(
        f"is_empty_patch = {is_empty} | "
        f"bg_ratio={bg_ratio:.4f}, "
        f"broad_stain_ratio={broad_stain_ratio:.6f}, "
        f"refined_stain_ratio={refined_stain_ratio:.6f}, "
        f"components={component_count}",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()



def should_exclude_patch(patch):
    return (
        is_empty_patch(patch) or
        #is_blurry_patch(patch, blur_threshold=50)
    )


if __name__ == "__main__":
    # empty_patch_cnt = 0

    # slide_path = "DS_B04R_04S"
    # base_path = f"./patches_tiles/{slide_path}"

    # if not os.path.isdir(base_path):
    #     raise FileNotFoundError(f"Patch folder not found: {base_path}")

    # excluded_dir = f"{base_path}/excluded"
    # os.makedirs(excluded_dir, exist_ok=True)

    # for filename in os.listdir(base_path):
    #     if filename.endswith(".png"):
    #         patch_path = os.path.join(base_path, filename)
    #         patch = cv2.imread(patch_path)

    #         if patch is None:
    #             print(f"Skipping unreadable image: {patch_path}")
    #             continue

    #         if is_empty_patch(patch, visualize=False):
    #             empty_patch_cnt += 1
    #             excluded_path = os.path.join(excluded_dir, filename)
    #             os.replace(patch_path, excluded_path)

    # print(f"Total empty patches: {empty_patch_cnt}")

    # Example usage with a sample patch
    patch_path = r"C:\Users\USER\Desktop\TAMU\AI-bile\patches_tiles\DS_B04R_04S\DS_B04R_04S_patch_78950_124160.png"
    patch = cv2.imread(patch_path)
    is_empty = is_empty_patch(patch, visualize=True)
