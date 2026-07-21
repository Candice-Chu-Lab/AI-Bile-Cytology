from pathlib import Path
import re
from pathlib import Path
import pandas as pd


# INPUT_FOLDER = Path(r"./Annotation_Positive_Selected/cross-validation_master_split")


LABEL_COLUMNS = [
    "Present",
    "Not-present",
    "Rods",
    "Cocci",
    "Yeast",
    "Few",
    "Moderate",
    "Abundant",
]


def parse_filename(file_path: Path) -> tuple[int, str]:
    """
    Expected filenames:
    fold_0_train.csv
    fold_0_val.csv
    fold_1_train.csv
    fold_1_val.csv
    """
    match = re.fullmatch(
        r"fold_(\d+)_(train|val)",
        file_path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Unexpected filename: {file_path.name}")

    fold_number = int(match.group(1))
    split_name = match.group(2).lower()

    return fold_number, split_name


def create_master_summary(INPUT_FOLDER) -> None:
    OUTPUT_FILE = Path(INPUT_FOLDER / "master_fold_summary.xlsx")
    files = sorted(INPUT_FOLDER.glob("fold_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No fold CSV files found in: {INPUT_FOLDER}"
        )

    summary_rows = []
    combined_frames = []

    for file_path in files:
        try:
            fold_number, split_name = parse_filename(file_path)
        except ValueError:
            print(f"Skipping unexpected file: {file_path.name}")
            continue

        df = pd.read_csv(file_path)

        missing_columns = [
            column
            for column in LABEL_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing columns: "
                f"{missing_columns}"
            )

        # Convert label columns to numeric 0/1 values.
        df[LABEL_COLUMNS] = (
            df[LABEL_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )

        summary_row = {
            "fold": fold_number,
            "split": split_name,
            "file_name": file_path.name,
            "total_images": len(df),
        }

        # Sum each binary label column.
        for column in LABEL_COLUMNS:
            summary_row[column] = int(df[column].sum())

        summary_rows.append(summary_row)

        # Store all rows for the combined-data sheet.
        current_df = df.copy()
        current_df.insert(0, "fold", fold_number)
        current_df.insert(1, "split", split_name)
        current_df.insert(2, "source_file", file_path.name)

        combined_frames.append(current_df)

    if not summary_rows:
        raise ValueError("No valid fold CSV files were processed.")

    summary_df = pd.DataFrame(summary_rows)

    split_order = {
        "train": 0,
        "val": 1,
    }

    summary_df["split_order"] = summary_df["split"].map(split_order)

    summary_df = (
        summary_df
        .sort_values(["fold", "split_order"])
        .drop(columns="split_order")
        .reset_index(drop=True)
    )

    combined_df = pd.concat(
        combined_frames,
        ignore_index=True,
    )

    summary_by_fold = (
        summary_df
        .groupby("fold", as_index=False)[
            ["total_images", *LABEL_COLUMNS]
        ]
        .sum()
    )

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:
        summary_df.to_excel(
            writer,
            sheet_name="Fold Summary",
            index=False,
        )

        summary_by_fold.to_excel(
            writer,
            sheet_name="Combined Train Val",
            index=False,
        )

        combined_df.to_excel(
            writer,
            sheet_name="All CSV Rows",
            index=False,
        )

        # Improve readability.
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value))
                    if cell.value is not None
                    else 0
                    for cell in cells
                )

                worksheet.column_dimensions[
                    cells[0].column_letter
                ].width = min(max_length + 2, 35)

    # check in the specific column
    result = (summary_df[LABEL_COLUMNS] > 0).all().all()

    print(result)  # True or False


    print(f"Master Excel file saved to:\n{OUTPUT_FILE}")

    return result


if __name__ == "__main__":
    create_master_summary()