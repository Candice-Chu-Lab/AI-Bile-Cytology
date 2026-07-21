import os 
import pandas as pd
import shutil

# algorithm: 

# Open the aggregated_labels for each case
# open the corresponding folder for each case
# if the paritcular image in that case is positive, copy it to the positive folder
# if the particular image in that case is negative, copy it to the negative folder


aggregated_labels_dir = r"D:\bile_sample\result\Annotation_Positive_Selected\aggregated_labels"
cases_dir = r"D:\bile_sample\result"

NEGATIVE_CASE = ['DS_A09R_12S', 'DS_A09R_11S', 'DS_B04R_03S', 'DS_A09R_09S', 'DS_A01R_20S', 'DS_B04R_07S', 'DS_A06R_06S']
NEGATIVE_CASE_ADDITIONAL = ['DS_A09R_08S', 'DS_A01R_17S']

SPECIAL_PROCESS = ['DS_A06R_04S']

# for filename in os.listdir(aggregated_labels_dir):
#     case_id = filename.removesuffix("_aggregated_labels.csv")
#     if case_id  not in SPECIAL_PROCESS:
#         # print(f"Skipping case {case_id} as it is not in the negative cases list.")
#         continue
#     # open the csv file and read the labels
#     # each row contains a filename with its label as columns true/false
#     case_labels = pd.read_csv(os.path.join(aggregated_labels_dir, filename))
#     #print(f"Processing case {case_id} with {len(case_labels)} images...")
#     case_folder = os.path.join(cases_dir, case_id)
#     # create positive and negative folders if they don't exist
#     positive_folder = os.path.join(case_folder, "positive")
#     negative_folder = os.path.join(case_folder, "negative")
#     os.makedirs(positive_folder, exist_ok=True)
#     os.makedirs(negative_folder, exist_ok=True)
#     print(f"Processing case {case_id} with {len(case_labels)} images...")

#     for index, row in case_labels.iterrows():
#         image_filename_json = row["slide_name"]
#         image_filename_jpg = image_filename_json.replace(".json", ".png")

#         is_positive = bool(row["Present"])
#         # print(f"Image {image_filename_jpg} is {'positive' if is_positive else 'negative'}...")
#         src_image_path = os.path.join(case_folder, image_filename_jpg)
#         if is_positive:
#             dst_image_path = os.path.join(positive_folder, image_filename_jpg)
#         else:
#             dst_image_path = os.path.join(negative_folder, image_filename_jpg)

#         # copy the image to the corresponding folder
#         if os.path.exists(src_image_path):
#             shutil.copy2(src_image_path, dst_image_path)



# check the positive and negative folders for each case to see if they contain the correct number of images

csv_result = []
for filename in os.listdir(aggregated_labels_dir):
    case_id = filename.removesuffix("_aggregated_labels.csv")
    case_folder = os.path.join(cases_dir, case_id)
    positive_folder = os.path.join(case_folder, "positive")
    negative_folder = os.path.join(case_folder, "negative")
    # count the positive and negative images in the folders
    positive_count = len(os.listdir(positive_folder))
    negative_count = len(os.listdir(negative_folder))
    csv_result.append([case_id, positive_count, negative_count])

df_result = pd.DataFrame(csv_result, columns=["case_id", "positive_count", "negative_count"])
df_result.to_csv(os.path.join(cases_dir, "positive_negative_counts.csv"), index=False)

