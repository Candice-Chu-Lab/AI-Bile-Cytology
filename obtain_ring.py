import os

import openslide
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_inner_ring_location(
    slide_path,
    thumbnail_size=(2000, 2000),
    lower_green=(35, 25, 25),
    upper_green=(95, 255, 255),
    kernel_size=5,
    min_area=1000,
    min_circularity=0.5,
    show_debug=False,
):
    slide = openslide.OpenSlide(slide_path)
    base_name = os.path.basename(slide_path).split('.')[0]
    slide_w, slide_h = slide.dimensions

    thumb = slide.get_thumbnail(thumbnail_size).convert("RGB")
    thumb_np = np.array(thumb)
    thumb_h, thumb_w = thumb_np.shape[:2]

    hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
    ring_mask = cv2.inRange(hsv, np.array(lower_green), np.array(upper_green))

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circularity = 4 * np.pi * area / (peri * peri)
        score = area * circularity

        if circularity > min_circularity and score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        raise RuntimeError("No suitable outer ring contour found. Try adjusting thresholds.")

    filled_disk = np.zeros_like(ring_mask)
    cv2.drawContours(filled_disk, [best_cnt], -1, 255, thickness=-1)

    inner_region = cv2.subtract(filled_disk, ring_mask)
    inner_region = cv2.morphologyEx(inner_region, cv2.MORPH_OPEN, kernel)
    inner_region = cv2.morphologyEx(inner_region, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component in the inner region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inner_region, connectivity=8)
    clean_inner = np.zeros_like(inner_region)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_inner[labels == largest_label] = 255
    else:
        raise RuntimeError("No inner hole found after connected-components filtering.")

    inner_contours, _ = cv2.findContours(clean_inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not inner_contours:
        raise RuntimeError("No inner contour found after cleaning.")

    best_inner = max(inner_contours, key=cv2.contourArea)

    (cx_inner, cy_inner), inner_radius = cv2.minEnclosingCircle(best_inner)
    cx_inner, cy_inner = int(cx_inner), int(cy_inner)
    inner_radius = int(inner_radius)

    scale_x = slide_w / float(thumb_w)
    scale_y = slide_h / float(thumb_h)

    inner_center_slide = (int(cx_inner * scale_x), int(cy_inner * scale_y))
    inner_radius_slide = int(inner_radius * (scale_x + scale_y) / 2.0)

    # result = {
    #     "inner_center_thumbnail": (cx_inner, cy_inner),
    #     "inner_radius_thumbnail": inner_radius,
    #     "inner_center_slide": inner_center_slide,
    #     "inner_radius_slide": inner_radius_slide,
    #     "inner_contour_thumbnail": best_inner,
    #     "inner_mask_thumbnail": clean_inner,
    #     "ring_mask_thumbnail": ring_mask,
    #     "thumbnail_rgb": thumb_np,
    #     "thumbnail_shape": (thumb_h, thumb_w),
    #     "slide_shape": (slide_h, slide_w),
    # }
    filled_inner = np.zeros_like(clean_inner)
    cv2.drawContours(filled_inner, [best_inner], -1, 255, thickness=-1)

    result = {
        "inner_center_thumbnail": (cx_inner, cy_inner),
        "inner_radius_thumbnail": inner_radius,
        "inner_center_slide": inner_center_slide,
        "inner_radius_slide": inner_radius_slide,
        "inner_contour_thumbnail": best_inner,
        "inner_mask_thumbnail": filled_inner,
        "ring_mask_thumbnail": ring_mask,
        "thumbnail_rgb": thumb_np,
        "thumbnail_shape": (thumb_h, thumb_w),
        "slide_shape": (slide_h, slide_w),
    }

    if show_debug:
        overlay = thumb_np.copy()
        cv2.drawContours(overlay, [best_cnt], -1, (255, 0, 0), 2)
        cv2.drawContours(overlay, [best_inner], -1, (0, 0, 255), 2)
        cv2.circle(overlay, (cx_inner, cy_inner), 4, (0, 255, 255), -1)

        fig, axes = plt.subplots(1, 5, figsize=(22, 8))

        axes[0].imshow(thumb_np)
        axes[0].set_title("Thumbnail")
        axes[0].axis("off")

        axes[1].imshow(ring_mask, cmap="gray")
        axes[1].set_title("Ring Mask")
        axes[1].axis("off")

        axes[2].imshow(filled_disk, cmap="gray")
        axes[2].set_title("Filled Outer Disk")
        axes[2].axis("off")

        axes[3].imshow(clean_inner, cmap="gray")
        axes[3].set_title("Clean Inner Hole")
        axes[3].axis("off")

        axes[4].imshow(overlay)
        axes[4].set_title("Outer = Red, Inner = Blue")
        axes[4].axis("off")

        plt.tight_layout()
        os.makedirs(f"outputs/{base_name}", exist_ok=True)
        plt.savefig(f"outputs/{base_name}/ring_detection_debug.png")

    return result


# if __name__ == "__main__":
#     ring_info = detect_inner_ring_location("DS_A09R_01S.mrxs", show_debug=True)
#     print("Inner center (thumbnail):", ring_info["inner_center_thumbnail"])
#     print("Inner radius (thumbnail):", ring_info["inner_radius_thumbnail"])
#     print("Inner center (slide):", ring_info["inner_center_slide"])
#     print("Inner radius (slide):", ring_info["inner_radius_slide"])