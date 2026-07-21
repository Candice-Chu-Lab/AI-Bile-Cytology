# Check conducted
import dataclasses
import json
import os
import tyro
from dataclasses import dataclass
import pandas as pd
import shutil

@dataclass
class jsonArgs:
    slides_path: str
    multiple_slides: bool = False


def check_all_json_exists(slides_path, label_path):
    # Use sets to ignore order
    slide_files = {f.removesuffix(".png") for f in os.listdir(slides_path) if f.endswith(".png")}
    label_files = {f.removesuffix(".json") for f in os.listdir(label_path) if f.endswith(".json")}
    
    return slide_files == label_files  # All slides have labels AND all labels have slides


def check_json_content(label_path):
    print("first several entries of the JSON files:", os.listdir(label_path)[:5])
    error_entry = []
    filter = [f for f in os.listdir(label_path) if f.endswith(".json")]
    print("========  logs for ", label_path, "  ========")
    print("a total of ", len(filter), " JSON files to check.")
    for filename in filter:
        with open(os.path.join(label_path, filename), "r") as f:
            data = json.load(f)
            flags = data.get("flags", {})


            is_not_present = flags.get("Not-present", False)
            is_not_usable = flags.get("Not-usable", False)
            # Rule 1: If Not-present or Not-Usable, no other flags should be true
            if is_not_present or is_not_usable:
                other_flags = ["Present", "Rods", "Cocci", "Yeast", "Few", "Moderate", "Abundant"]
                # Also ensure Not-present and Not-Usable are not both true
                if is_not_present and is_not_usable:
                    print(f"Error in {filename}: 'Not-present' and 'Not-Usable' cannot both be true.")
                    error_entry.append(filename)
                elif any(flags.get(label, False) for label in other_flags):
                    print(f"Error in {filename}: If 'Not-present' or 'Not-Usable' is true, no other flags should be true.")
                    error_entry.append(filename)
                continue  # Skip further checks for this file

            # Rule 2: Exactly one of Rods, Cocci, Yeast must be true
            type_flags = [flags.get("Rods", False), flags.get("Cocci", False), flags.get("Yeast", False)]
            if sum(type_flags) != 1:
                print(f"Error in {filename}: Exactly one of 'Rods', 'Cocci', or 'Yeast' must be true.")
                error_entry.append(filename)

            # Rule 3: Exactly one of Few, Moderate, Abundant must be true
            quantity_flags = [flags.get("Few", False), flags.get("Moderate", False), flags.get("Abundant", False)]
            if sum(quantity_flags) != 1:
                print(f"Error in {filename}: Exactly one of 'Few', 'Moderate', or 'Abundant' must be true.")
                error_entry.append(filename)

            # Rule 4: Present must be true if we reach here
            if not flags.get("Present", False):
                print(f"Error in {filename}: 'Present' must be true when specifying type and quantity.")
                error_entry.append(filename)

    return error_entry

# add a function to calculate total count of positive/negative cases based on the JSON files, which can be used for stratification in the next step
# save to csv so that stratification can be used
def calculate_total_count(label_path):
    columns = ["Present",'Not-present','Not-usable', "Rods", "Cocci", "Yeast", "Few", "Moderate", "Abundant"]
    counts = {col: 0 for col in columns}
    labels = os.listdir(label_path)
    for filename in labels:
        if filename.endswith(".json"):
            with open(os.path.join(label_path, filename), "r") as f:
                data = json.load(f)
                flags = data.get("flags", {})
                for col in columns:
                    if flags.get(col, True):
                        counts[col] += 1
    return counts

# note: the action is not recoverable, it will directly delete the unsable JSON files
# but it will save a copy of the unsable JSON files in a new folder called "filtered_json" in the slide_path
def filter_out_unusable(slide_path):
    # Create a new folder for filtered JSON files
    print("========  filter logs for ", slide_path, "  ========")
    filtered_folder = os.path.join(slide_path, "filtered_json")
    os.makedirs(filtered_folder, exist_ok=True)
    cnt = 0

    # Iterate through all JSON files in the slide_path
    for filename in os.listdir(slide_path):
        if filename.endswith(".json"):
            with open(os.path.join(slide_path, filename), "r") as f:
                data = json.load(f)
                flags = data.get("flags", {})

            # Check if the JSON contains the file not usable
            is_not_usable = flags.get("Not-usable", True)

            if is_not_usable:
                print(f"Excluding {filename} as it is marked 'Not-usable'.")
                # Copy the unsable JSON file to the filtered folder
                shutil.copy2(os.path.join(slide_path, filename), os.path.join(filtered_folder, filename))
                # removing the unsable JSON file from the original folder
                os.remove(os.path.join(slide_path, filename))
                cnt+= 1
    print(f"Total {cnt} unsable JSON files filtered out of {len(os.listdir(slide_path))} total JSON files.")
        

# python post_process/json_check.py --slides_path Annotation_Positive_Selected/
if __name__ == "__main__":
    args = tyro.cli(jsonArgs)
    slide_list = os.listdir(args.slides_path)
    # exclude any non-directory items
    slide_list = [i for i in slide_list if os.path.isdir(os.path.join(args.slides_path, i))]
    # The base folder contains lots of cases
    count_dict = {}
    for i in slide_list:
        slide_path = os.path.join(args.slides_path, i)
        # step_1_check = check_all_json_exists(args.slides_path, args.slides_path)
        # if step_1_check:
        #     print("All JSON files exist for the corresponding slide files.")
        # else:
        #     raise FileNotFoundError("Mismatch between slide files and JSON label files. Please check the directories.")

        # if not os.path.isdir(slide_path):
        #     continue
        step_2_check = check_json_content(slide_path)
        if not step_2_check:
            print("All JSON files passed content checks.")

        step_3 = filter_out_unusable(slide_path)

        individual_cnt = calculate_total_count(slide_path)
        count_dict[i] = individual_cnt
    # Save the count_dict to a CSV file
    df = pd.DataFrame.from_dict(count_dict, orient='index')
    df.to_excel(os.path.join("Annotation_Positive_Selected", "total_count.xlsx"))
    
