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
            # Perform content checks on 'data' as needed
            flags = data.get("flags", {})
            # TODO: revise based on the final decision on labels
            # one of rod, cone must be true, but not both
            if not (flags.get("rod", False) ^ flags.get("cone", False)):
                print(f"Error in {filename}: Both 'rod' and 'cone' flags cannot be true or both cannot be false.")
                error_entry.append(filename)

            # Exactly one of small, medium, large must be true
            size_flags = [flags.get("small", False), flags.get("medium", False), flags.get("large", False)]
            if sum(size_flags) != 1:
                print(f"Error in {filename}: Exactly one of 'small', 'medium', or 'large' flags must be true.")
                error_entry.append(filename)

    return error_entry

# TODO: add a function to calculate total count of positive/negative cases based on the JSON files, which can be used for stratification in the next step
# save to csv so that stratification can be used
def calculate_total_count(label_path):
    pass


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
    
