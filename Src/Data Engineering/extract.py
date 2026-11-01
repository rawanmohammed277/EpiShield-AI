from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SRC_DIR / "Data"

def extract_data():
    files = {
        "regioni": DATA_DIR / "regioni.csv",
        "nazionale": DATA_DIR / "nazionale.csv",
        "popolazione": DATA_DIR / "popolazione.csv",
        "note": DATA_DIR / "note.csv"
    }

    data = {}

    for name, file_path in files.items():
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        data[name] = df

        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")

    return data


if __name__ == "__main__":
    extract_data()