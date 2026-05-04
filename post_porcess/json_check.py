# Check conducted
import json
import os

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





slide_path = "./patches_tiles/slides/"
label_path = "./patches_tiles/slides/"
step_1_check = check_all_json_exists(slide_path, label_path)
if step_1_check:
    print("All JSON files exist for the corresponding slide files.")

step_2_check = check_json_content(label_path)
if not step_2_check:
    print("All JSON files passed content checks.")