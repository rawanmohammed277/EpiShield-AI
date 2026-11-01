import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "Data"
def extract_data():

    files = {
        "regioni": BASE_DIR / "regioni.csv",
        "popolazione": BASE_DIR / "popolazione.csv",
        "note": BASE_DIR / "note.csv",
        "nazionale": BASE_DIR / "nazionale.csv",
    }

    dataframes = {}

    for name, path in files.items():

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"File is empty: {path}")

        dataframes[name] = df

        print(f"{name}: Loaded successfully")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("-" * 40)

    return dataframes


if __name__ == "__main__":
    data = extract_data()