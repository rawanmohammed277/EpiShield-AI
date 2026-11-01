
import pandas as pd
import os

def extract_data():

    files = {
        "regioni": "data/regioni.csv",
        "popolazione": "data/popolazione.csv",
        "note": "data/note.csv",
        "nazionale": "data/nazionale.csv"
    }

    dataframes = {}

    for name, path in files.items():

        if not os.path.exists(path):
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
