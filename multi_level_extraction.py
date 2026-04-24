# import math
# import openslide
# import numpy as np
# import matplotlib.pyplot as plt


# def um_to_pixels(slide, x_um, y_um):
#     mpp_x = float(slide.properties.get("openslide.mpp-x", 0.25))
#     mpp_y = float(slide.properties.get("openslide.mpp-y", 0.25))

#     print(f"MPP X: {mpp_x}, MPP Y: {mpp_y}")

#     x_pixels = int(round(x_um / mpp_x))
#     y_pixels = int(round(y_um / mpp_y))
#     return x_pixels, y_pixels


# def extract_patches(slide, x_center, y_center, levels_to_use, patch_size=512, save_path="multi_level_patches.png"):
#     patches = {}

#     for level in levels_to_use:
#         ds = float(slide.level_downsamples[level])

#         # Convert half patch size into level-0 coordinate system
#         half_size = int(round((patch_size * ds) / 2))

#         x0 = max(0, int(round(x_center - half_size)))
#         y0 = max(0, int(round(y_center - half_size)))

#         patch = slide.read_region((x0, y0), level, (patch_size, patch_size))
#         patch = np.array(patch.convert("RGB"))
#         patches[level] = patch

#     print({k: v.shape for k, v in patches.items()})

#     # Plot grid automatically
#     n = len(levels_to_use)
#     ncols = 5
#     nrows = math.ceil(n / ncols)

#     fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

#     # Make axes always iterable
#     if nrows == 1:
#         axes = np.array([axes])
#     axes = axes.reshape(nrows, ncols)

#     for i, level in enumerate(levels_to_use):
#         row = i // ncols
#         col = i % ncols

#         axes[row, col].imshow(patches[level])
#         axes[row, col].set_title(
#             f"Level {level}\nDS={slide.level_downsamples[level]:.2f}"
#         )
#         axes[row, col].axis("off")

#     # Hide unused axes
#     for j in range(n, nrows * ncols):
#         row = j // ncols
#         col = j % ncols
#         axes[row, col].axis("off")

#     plt.suptitle(f"Centered patches at ({x_center}, {y_center}) in level-0 pixels", fontsize=16)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.show()

#     return patches


import math
import openslide
import numpy as np
import matplotlib.pyplot as plt
import cv2


# def um_to_pixels(slide, x_um, y_um):
#     mpp_x = float(slide.properties.get("openslide.mpp-x", 0.25))
#     mpp_y = float(slide.properties.get("openslide.mpp-y", 0.25))

#     print(f"MPP X: {mpp_x}, MPP Y: {mpp_y}")

#     x_pixels = int(round(x_um / mpp_x))
#     y_pixels = int(round(y_um / mpp_y))
#     return x_pixels, y_pixels

def um_to_pixels(slide, x_um, y_um):
    mpp_x = float(slide.properties["openslide.mpp-x"])
    mpp_y = float(slide.properties["openslide.mpp-y"])

    bounds_x = int(slide.properties.get("openslide.bounds-x", 0))
    bounds_y = int(slide.properties.get("openslide.bounds-y", 0))

    x_pixels = int(round(x_um / mpp_x)) + bounds_x
    y_pixels = int(round(y_um / mpp_y)) + bounds_y

    return x_pixels, y_pixels

def extract_patches(
    slide,
    x_center,
    y_center,
    levels_to_use,
    patch_size=512,
    show_center=True,
    save_path="multi_level_patches.png"
):
    patches = {}

    for level in levels_to_use:
        ds = float(slide.level_downsamples[level])

        half_size = int(round((patch_size * ds) / 2))

        x0 = max(0, int(round(x_center - half_size)))
        y0 = max(0, int(round(y_center - half_size)))

        patch = slide.read_region((x0, y0), level, (patch_size, patch_size))
        patch = np.array(patch.convert("RGB"))

        # # 🔴 draw center circle
        # if show_center:
        #     h, w = patch.shape[:2]
        #     cx, cy = w // 2, h // 2
        #     cv2.circle(patch, (cx, cy), 10, (255, 0, 0), 2)  # red circle

        patches[level] = patch

    # print({k: v.shape for k, v in patches.items()})

    # -------- Plot grid --------
    n = len(levels_to_use)
    ncols = 5
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for i, level in enumerate(levels_to_use):
        row = i // ncols
        col = i % ncols

        axes[row, col].imshow(patches[level])
        axes[row, col].set_title(
            f"Level {level}\nDS={slide.level_downsamples[level]:.2f}"
        )
        axes[row, col].axis("off")

    # hide unused
    for j in range(n, nrows * ncols):
        row = j // ncols
        col = j % ncols
        axes[row, col].axis("off")

    plt.suptitle(f"Centered patches at ({x_center}, {y_center})", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    #plt.show()


    # save every if level in patches, limit to level 0 and 1
    for level, patch in patches.items():
        if level in [0, 1]:
            cv2.imwrite(f"patch_level_{level}_{x_center}_{y_center}.png", cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

    return patches


if __name__ == "__main__":
    slide_path = "DS_A09R_01S.mrxs"
    x_um, y_um = 12656, 25799
    levels_to_use = list(range(10))

    slide = openslide.OpenSlide(slide_path)

    x_px, y_px = um_to_pixels(slide, x_um, y_um)
    patches = extract_patches(slide, x_px, y_px, levels_to_use, patch_size=512)