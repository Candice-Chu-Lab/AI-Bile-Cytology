import cv2
import numpy as np
import matplotlib.pyplot as plt


def analyze_patch_filter(
    patch,
    bg_threshold=220,
    bg_ratio_threshold=0.90,
    stain_ratio_threshold=0.0003,
    min_area=10,
    max_area=300,
    min_aspect_ratio=2.0,
    visualize=True
):
    """
    Analyze whether a patch is empty, and optionally visualize each filtering step.

    Returns:
        result: dict containing ratios, component count, boxes, masks, and is_empty_patch.
    """

    # -------------------------------------------------
    # 1. Original image
    # -------------------------------------------------
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------
    # 2. Background ratio
    # -------------------------------------------------
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    background_mask = gray > bg_threshold
    bg_ratio = background_mask.sum() / background_mask.size

    # -------------------------------------------------
    # 3. HSV conversion
    # -------------------------------------------------
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # -------------------------------------------------
    # 4. Original stained pixel mask
    #    This is used for stain_ratio.
    # -------------------------------------------------
    stained_mask_original = (
        (s > 40) &
        (v < 230) &
        (h > 90) & (h < 170)
    )

    stain_ratio = stained_mask_original.sum() / stained_mask_original.size

    # -------------------------------------------------
    # 5. Refined component mask
    #    This is used for connected components.
    # -------------------------------------------------
    component_mask = (
        (s > 50) &
        (v < 180) &
        (h > 90) & (h < 170)
    ).astype(np.uint8)

    # Remove tiny isolated noise
    kernel = np.ones((2, 2), np.uint8)
    component_mask_opened = cv2.morphologyEx(
        component_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # -------------------------------------------------
    # 6. Connected components
    # -------------------------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        component_mask_opened
    )

    kept_boxes = []
    rejected_boxes = []

    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]

        aspect_ratio = max(w, h_box) / max(1, min(w, h_box))

        component_info = {
            "x": x,
            "y": y,
            "w": w,
            "h": h_box,
            "area": area,
            "aspect_ratio": aspect_ratio,
        }

        if area < min_area:
            component_info["reason"] = "area too small"
            rejected_boxes.append(component_info)
            continue

        if area > max_area:
            component_info["reason"] = "area too large"
            rejected_boxes.append(component_info)
            continue

        if aspect_ratio < min_aspect_ratio:
            component_info["reason"] = "not elongated enough"
            rejected_boxes.append(component_info)
            continue

        kept_boxes.append(component_info)

    component_count = len(kept_boxes)

    # -------------------------------------------------
    # 7. Final empty-patch decision
    # -------------------------------------------------
    is_empty = (
        bg_ratio > bg_ratio_threshold and
        stain_ratio < stain_ratio_threshold and
        component_count == 0
    )

    # -------------------------------------------------
    # 8. Visualization
    # -------------------------------------------------
    if visualize:
        kept_vis = patch_rgb.copy()
        rejected_vis = patch_rgb.copy()

        # Kept components: red boxes
        for box in kept_boxes:
            x, y, w, h_box = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(
                kept_vis,
                (x, y),
                (x + w, y + h_box),
                (255, 0, 0),
                1
            )

        # Rejected components: yellow boxes
        for box in rejected_boxes:
            x, y, w, h_box = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(
                rejected_vis,
                (x, y),
                (x + w, y + h_box),
                (255, 255, 0),
                1
            )

        plt.figure(figsize=(18, 12))

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
        plt.imshow(stained_mask_original, cmap="gray")
        plt.title(f"5. Original stained mask\nstain_ratio={stain_ratio:.6f}")
        plt.axis("off")

        plt.subplot(2, 4, 6)
        plt.imshow(component_mask, cmap="gray")
        plt.title("6. Refined component mask\nbefore morphology")
        plt.axis("off")

        plt.subplot(2, 4, 7)
        plt.imshow(component_mask_opened, cmap="gray")
        plt.title("7. Refined mask\nafter morphology")
        plt.axis("off")

        plt.subplot(2, 4, 8)
        plt.imshow(kept_vis)
        plt.title(f"8. Kept components\ncount={component_count}")
        plt.axis("off")

        plt.suptitle(
            f"is_empty_patch = {is_empty} | "
            f"bg_ratio={bg_ratio:.4f}, "
            f"stain_ratio={stain_ratio:.6f}, "
            f"components={component_count}",
            fontsize=16
        )

        plt.tight_layout()
        plt.show()

        # Optional second figure: rejected boxes
        plt.figure(figsize=(7, 7))
        plt.imshow(rejected_vis)
        plt.title(f"Rejected components: {len(rejected_boxes)}")
        plt.axis("off")
        plt.show()

    # -------------------------------------------------
    # 9. Return everything useful for debugging
    # -------------------------------------------------
    result = {
        "is_empty_patch": is_empty,
        "bg_ratio": bg_ratio,
        "stain_ratio": stain_ratio,
        "component_count": component_count,
        "kept_boxes": kept_boxes,
        "rejected_boxes": rejected_boxes,
        "gray": gray,
        "background_mask": background_mask,
        "hue": h,
        "saturation": s,
        "value": v,
        "stained_mask_original": stained_mask_original,
        "component_mask": component_mask,
        "component_mask_opened": component_mask_opened,
    }

    return result


if __name__ == "__main__":
    # Example usage with a sample patch
    patch_path = r"C:\Users\USER\Desktop\TAMU\AI-bile\patches_tiles\DS_B04R_04S\DS_B04R_04S_patch_63060_125626.png"
    patch = cv2.imread(patch_path)

    result = analyze_patch_filter(patch, visualize=True)