import os
import shutil
import pandas as pd

from pathlib import Path
import pandas as pd

import os
import shutil


summary_root = r"D:\bile_sample\result\Annotation_Positive_Selected\summary\cross-validation"
source_root = r"D:\bile_sample\result"
result_root = r"D:\bile_sample\training_data"
copy_folder = False 


def copy_slide_files(slide_names, source_root, destination_root, source_category):
    for slide_name in slide_names:
        print(f"Processing slide: {slide_name} from category: {source_category}")
        src_folder = os.path.join(source_root, slide_name, source_category)
        dst_folder = os.path.join(destination_root, slide_name)

        if not os.path.isdir(src_folder):
            print(f"Missing folder: {src_folder}")
            continue

        os.makedirs(dst_folder, exist_ok=True)

        for filename in os.listdir(src_folder):
            src_file = os.path.join(src_folder, filename)
            dst_file = os.path.join(dst_folder, filename)

            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)


def calculate_folder_positive_negative_counts(folder_path):
    positive_count = 0
    negative_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                if "Present" in root:
                    positive_count += 1
                elif "Not-present" in root:
                    negative_count += 1

    return positive_count, negative_count





def count_files_by_slide(
    training_root: str,
    output_csv: str = "file_counts_by_slide.csv",
    aggregate_csv: str = "file_counts_aggregate.csv",
):
    """
    Expected folder structure:

    training_root/
        fold_0/
            train/
                Present/
                    slide_1/
                    slide_2/
                Not-present/
                    slide_1/
                    slide_2/
            val/
                Present/
                Not-present/
        fold_1/
        fold_2/
    """

    training_root = Path(training_root)
    records = []

    for fold_folder in sorted(training_root.glob("fold_*")):
        if not fold_folder.is_dir():
            continue

        for split in ["train", "val"]:
            split_folder = fold_folder / split

            for label in ["Present", "Not-present"]:
                label_folder = split_folder / label

                if not label_folder.exists():
                    print(f"Folder not found: {label_folder}")
                    continue

                # Each folder inside Present/Not-present is a slide folder
                for slide_folder in sorted(label_folder.iterdir()):
                    if not slide_folder.is_dir():
                        continue

                    # Count all files recursively inside the slide folder
                    file_count = sum(
                        1
                        for path in slide_folder.rglob("*")
                        if path.is_file()
                    )

                    records.append({
                        "fold": fold_folder.name,
                        "split": split,
                        "label": label,
                        "slide_name": slide_folder.name,
                        "file_count": file_count,
                    })

    detail_df = pd.DataFrame(records)

    if detail_df.empty:
        print("No files were found.")
        return detail_df, pd.DataFrame()

    # Export detailed count for every slide folder
    detail_df.to_csv(output_csv, index=False)

    # Aggregate by fold, split, and label
    aggregate_df = (
        detail_df
        .groupby(["fold", "split", "label"], as_index=False)
        .agg(
            number_of_slides=("slide_name", "nunique"),
            total_files=("file_count", "sum"),
        )
    )

    aggregate_df.to_csv(aggregate_csv, index=False)

    print(f"Detailed counts saved to: {output_csv}")
    print(f"Aggregate counts saved to: {aggregate_csv}")

    return detail_df, aggregate_df


from pathlib import Path
import shutil


def flatten_class_folders(train_folder, copy_files=True):
    """
    Flatten:
        train/Present/slide_name/image.jpg
    into:
        train/Present/slide_name_image.jpg

    The slide folder name is added to the filename to avoid collisions.
    """

    train_folder = Path(train_folder)

    for class_name in ["Present", "Not-present"]:
        class_folder = train_folder / class_name

        if not class_folder.is_dir():
            print(f"Missing folder: {class_folder}")
            continue

        copied_count = 0

        for slide_folder in class_folder.iterdir():
            if not slide_folder.is_dir():
                continue

            for source_file in slide_folder.rglob("*"):
                if not source_file.is_file():
                    continue

                # Prefix the filename with the slide folder name
                new_filename = f"{source_file.name}"
                destination_file = class_folder / new_filename

                # Handle duplicate filenames safely
                counter = 1
                while destination_file.exists():
                    new_filename = (
                        f"{slide_folder.name}_{source_file.stem}_{counter}"
                        f"{source_file.suffix}"
                    )
                    destination_file = class_folder / new_filename
                    counter += 1

                if copy_files:
                    shutil.copy2(source_file, destination_file)
                else:
                    shutil.move(str(source_file), str(destination_file))

                copied_count += 1

        action = "Copied" if copy_files else "Moved"
        print(f"{action} {copied_count} files into {class_folder}")


def remove_inner_folders(train_folder):
    train_folder = Path(train_folder)

    for class_name in ["Present", "Not-present"]:
        class_folder = train_folder / class_name

        if not class_folder.is_dir():
            continue

        for folder in class_folder.iterdir():
            if folder.is_dir():
                shutil.rmtree(folder)


def copy_folders(fold):
    fold_root = os.path.join(result_root, f"fold_{fold}")

    train_positive = os.path.join(fold_root, "train", "Present")
    train_negative = os.path.join(fold_root, "train", "Not-present")
    val_positive = os.path.join(fold_root, "val", "Present")
    val_negative = os.path.join(fold_root, "val", "Not-present")

    for folder in [
        train_positive,
        train_negative,
        val_positive,
        val_negative,
    ]:
        os.makedirs(folder, exist_ok=True)

    train_df = pd.read_csv(
        os.path.join(summary_root, f"fold_{fold}_train.csv")
    )
    val_df = pd.read_csv(
        os.path.join(summary_root, f"fold_{fold}_val.csv")
    )

    print("create training data for fold", fold)
    copy_slide_files(
        train_df["slide_name"],
        source_root,
        train_positive,
        "positive",
    )

    copy_slide_files(
        train_df["slide_name"],
        source_root,
        train_negative,
        "negative", 
    )

    print("create validation data for fold", fold)
    copy_slide_files(
        val_df["slide_name"],
        source_root,
        val_positive,
        "positive",
    )

    copy_slide_files(
        val_df["slide_name"],
        source_root,
        val_negative,
        "negative",
    )






for fold in range(0, 3):
    if copy_folder:
        copy_folders(fold)



    training_root = r"D:\bile_sample\training_data"

    # detail_df, aggregate_df = count_files_by_slide(
    #     training_root=training_root,
    #     output_csv=r"D:\bile_sample\training_data\file_counts_by_slide_fold2.csv",
    #     aggregate_csv=r"D:\bile_sample\training_data\file_counts_aggregate_fold2.csv",
    # )

    flatten_class_folders(
        train_folder=os.path.join(training_root, f"fold_{fold}", "train"),
        copy_files=False,
    )
    flatten_class_folders(
        train_folder=os.path.join(training_root, f"fold_{fold}", "val"),
        copy_files=False,
    )

    train_folder = os.path.join(training_root, f"fold_{fold}", "train")
    val_folder = os.path.join(training_root, f"fold_{fold}", "val")
    remove_inner_folders(train_folder)
    remove_inner_folders(val_folder)
    print(f"Removed inner folders for fold {fold}")