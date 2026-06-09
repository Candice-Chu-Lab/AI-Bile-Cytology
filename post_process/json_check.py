# Check conducted
import dataclasses
import json
import os
import tyro
from dataclasses import dataclass

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
    error_entry = []
    filter = [f for f in os.listdir(label_path) if f.endswith(".json")]
    for filename in filter:
        with open(os.path.join(label_path, filename), "r") as f:
            data = json.load(f)
            flags = data.get("flags", {})

            is_not_present = flags.get("Not-present", False)
            is_not_usable = flags.get("Not-Usable", False)

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

# TODO: add a function to calculate total count of positive/negative cases based on the JSON files, which can be used for stratification in the next step
# save to csv so that stratification can be used
def calculate_total_count(label_path):
    pass


# python post_process/json_check.py --slides_path DS_A09R_16S_first50_Annotation/
if __name__ == "__main__":
    args = tyro.cli(jsonArgs)
    step_1_check = check_all_json_exists(args.slides_path, args.slides_path)
    if step_1_check:
        print("All JSON files exist for the corresponding slide files.")
    else:
        raise FileNotFoundError("Mismatch between slide files and JSON label files. Please check the directories.")

    step_2_check = check_json_content(args.slides_path)
    if not step_2_check:
        print("All JSON files passed content checks.")

    calculate_total_count(args.slides_path)
    
