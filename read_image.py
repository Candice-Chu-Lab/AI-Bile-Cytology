import openslide

slide = openslide.OpenSlide("DS_A09R_01S.mrxs")
# print("Number of levels:", slide.level_count)
# for i in range(slide.level_count):
#     print(f"Level {i}:")
#     print("  Dimensions:", slide.level_dimensions[i])
#     print("  Downsample:", slide.level_downsamples[i])


# mpp_x = slide.properties.get("openslide.mpp-x")
# mpp_y = slide.properties.get("openslide.mpp-y")

# print("MPP X:", mpp_x)
# print("MPP Y:", mpp_y)



# base_mpp = float(slide.properties["openslide.mpp-x"])

# for level, ds in enumerate(slide.level_downsamples):
#     level_mpp = base_mpp * ds
#     print(f"Level {level}: MPP = {level_mpp:.4f}")




print(slide.properties.get("openslide.bounds-x"))
print(slide.properties.get("openslide.bounds-y"))
print(slide.properties.get("openslide.bounds-width"))
print(slide.properties.get("openslide.bounds-height"))