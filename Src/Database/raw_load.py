import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

# Add project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "Src"
DATA_DIR = SRC_DIR / "Data"

# Import database connection
sys.path.append(str(Path(__file__).resolve().parent))
from connection import engine


# CSV files and their corresponding SQL tables
RAW_TABLES = {
    "regioni": "RawRegionalData",
    "nazionale": "RawNationalData",
    "popolazione": "RawPopulationData",
    "note": "RawNotesData"
}


def load_raw_data():

    try:
        # Step 1: Read CSV files
        data = {}

        for file_name, table_name in RAW_TABLES.items():
            file_path = DATA_DIR / f"{file_name}.csv"

            df = pd.read_csv(
                file_path,
                dtype=str,
                keep_default_na=False
            )

            data[table_name] = df

            print(f"Extracted {file_name}: {len(df)} rows")

        # Step 2: Load data into Raw tables
        with engine.begin() as connection:

            for table_name, df in data.items():

                # Clear previous raw data
                connection.execute(
                    text(f"DELETE FROM dbo.[{table_name}]")
                )

                # Insert CSV data
                df.to_sql(
                    name=table_name,
                    con=connection,
                    schema="dbo",
                    if_exists="append",
                    index=False,
                    chunksize=1000
                )

                print(f"Loaded {len(df)} rows into {table_name}")

        print("\nRaw Data Loading Completed Successfully!")

    except Exception as error:
        print("\nRaw Data Loading Failed!")
        print(error)


if __name__ == "__main__":
    load_raw_data()