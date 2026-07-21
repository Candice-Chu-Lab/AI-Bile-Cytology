import os
import json
import pandas as pd

target_folder = "Annotation_Positive_Selected/"
save_folder = "Annotation_Positive_Selected/aggregated_labels/"
os.makedirs(save_folder, exist_ok=True)


def create_positive_case_csv():
    subfolders = [
        f
        for f in os.listdir(target_folder)
        if f.startswith("DS_")
        and os.path.isdir(os.path.join(target_folder, f))
    ]

    print(subfolders)

    # goal: for each subfolder, aggregate the labels from the JSON files into a single CSV file
    for subfolder in subfolders:
        subfolder_path = os.path.join(target_folder, subfolder)
        json_files = [
            f
            for f in os.listdir(subfolder_path)
            if f.endswith(".json") and os.path.isfile(os.path.join(subfolder_path, f))
        ]

        # Initialize a list to hold the aggregated data
        aggregated_data = []

        # Iterate through each JSON file and extract the relevant information
        for json_file in json_files:
            json_path = os.path.join(subfolder_path, json_file)
            with open(json_path, "r") as f:
                data = json.load(f)
                flags = data.get("flags", {})
                # Extract relevant information (e.g., slide name, flags)
                # "Rods", "Cocci", "Yeast", "Few", "Moderate", "Abundant"
                slide_name = data.get("slide_name", "")
                present_flag = flags.get("Present", False)
                not_present_flag = flags.get("Not-present", False)
                rods_flag = flags.get("Rods", False)
                cocci_flag = flags.get("Cocci", False)
                yeast_flag = flags.get("Yeast", False)
                few_flag = flags.get("Few", False)
                moderate_flag = flags.get("Moderate", False)
                abundant_flag = flags.get("Abundant", False)

                # Append the extracted information to the aggregated data list
                aggregated_data.append(
                    {
                        "slide_name": json_file,
                        "Present": present_flag,
                        "Not-present": not_present_flag,
                        "Rods": rods_flag,
                        "Cocci": cocci_flag,
                        "Yeast": yeast_flag,
                        "Few": few_flag,
                        "Moderate": moderate_flag,
                        "Abundant": abundant_flag,
                    }
                )

        # Convert the aggregated data into a DataFrame
        df_aggregated = pd.DataFrame(aggregated_data)

        # Save the aggregated DataFrame to a CSV file in the subfolder
        output_csv_path = os.path.join(save_folder, f"{subfolder}_aggregated_labels.csv")
        df_aggregated.to_csv(output_csv_path, index=False)
        print(f"Aggregated labels saved to {output_csv_path}")


def create_negative_case_csv():
    # create dummy CSV files for the negative cases with alll flags set to False, but the Not-Present is True
    NEGATIVE_CASE = ['DS_A09R_12S', 'DS_A09R_11S', 'DS_B04R_03S', 'DS_A09R_09S', 'DS_A01R_20S', 'DS_B04R_07S', 'DS_A06R_06S']
    NEGATIVE_CASE_ADDITIONAL = ['DS_A09R_08S', 'DS_A01R_17S']

    folder_location = r"D:\bile_sample\result"
    for case in NEGATIVE_CASE + NEGATIVE_CASE_ADDITIONAL:
        folder_path = os.path.join(folder_location, case, "selected")
        output_csv_path = os.path.join(save_folder, f"{case}_aggregated_labels.csv")
        aggregated_data = []
        for filename in os.listdir(folder_path):
            filename = filename.replace(".png", ".json")
            # Create a DataFrame with all flags set to False, but Not-Present set to True
            aggregated_data.append(
                
                    {
                        "slide_name": f"{filename}",
                        "Present": False,
                        "Not-present": True,
                        "Rods": False,
                        "Cocci": False,
                        "Yeast": False,
                        "Few": False,
                        "Moderate": False,
                        "Abundant": False,
                    }
                
            )

        df_negative = pd.DataFrame(aggregated_data)
        df_negative.to_csv(output_csv_path, index=False)
        print(f"Dummy aggregated labels saved to {output_csv_path}")


if __name__ == "__main__":
    #create_positive_case_csv()
    create_negative_case_csv()