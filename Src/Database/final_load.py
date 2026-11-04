from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "Src"

ETL_OUTPUT_DIR = SRC_DIR / "ETL Output"
DATABASE_DIR = SRC_DIR / "Database"

# IMPORTANT:
# Keep False on Doaa's computer.
# Set True ONLY on the teammate's computer where SQL Server exists.
RUN_DATABASE_LOAD = True

DATABASE_NAME = "EpiShieldDB"
SCHEMA_NAME = "dbo"

CHUNK_SIZE = 1000


# ============================================================
# 2. INPUT FILES
# ============================================================

FILE_PATHS = {
    "DimLocation": ETL_OUTPUT_DIR / "dim_location_load.csv",
    "DimDate": ETL_OUTPUT_DIR / "dim_date_load.csv",
    "FactPopulation": ETL_OUTPUT_DIR / "fact_population_staging.csv",
    "FactNational": ETL_OUTPUT_DIR / "fact_national_staging.csv",
    "FactRegional": ETL_OUTPUT_DIR / "fact_regional_staging.csv",
    "ReportNotes": ETL_OUTPUT_DIR / "report_notes_staging.csv",
}


# ============================================================
# 3. EXPECTED INPUT COLUMNS
# ============================================================

EXPECTED_COLUMNS = {
    "DimLocation": [
        "Region_Code",
        "Region_Name",
        "Region_Abbreviation",
        "Latitude",
        "Longitude",
        "NUTS_1_Code",
        "NUTS_1_Description",
        "NUTS_2_Code",
    ],

    "DimDate": [
        "Date",
    ],

    "FactPopulation": [
        "Region_Code",
        "Age_Range",
        "Total_Male",
        "Total_Female",
        "Total_Population",
    ],

    "FactNational": [
        "Date_Key",
        "symptomatic_hospitalized",
        "intensive_care",
        "total_hospitalized",
        "home_isolation",
        "total_positive",
        "positive_variation",
        "new_positive_cases",
        "recovered",
        "deaths",
        "suspected_diagnosis_cases",
        "screening_cases",
        "total_cases",
        "total_tests",
        "tested_cases",
        "intensive_care_admissions",
        "molecular_positive_cases",
        "rapid_antigen_positive_cases",
        "molecular_tests",
        "rapid_antigen_tests",
    ],

    "FactRegional": [
        "Region_Code",
        "Date_Key",
        "symptomatic_hospitalized",
        "intensive_care",
        "total_hospitalized",
        "home_isolation",
        "total_positive",
        "positive_variation",
        "new_positive_cases",
        "recovered",
        "deaths",
        "suspected_diagnosis_cases",
        "screening_cases",
        "total_cases",
        "total_tests",
        "tested_cases",
        "intensive_care_admissions",
        "molecular_positive_cases",
        "rapid_antigen_positive_cases",
        "molecular_tests",
        "rapid_antigen_tests",
    ],

    "ReportNotes": [
        "Date",
        "Region_Code",
        "source_type",
        "Note",
    ],
}


# ============================================================
# 4. SQL COLUMN MAPPINGS
# ============================================================

NATIONAL_COLUMN_MAPPING = {
    "symptomatic_hospitalized": "Hospitalized_with_Symptoms",
    "intensive_care": "Intensive_Care_ICU",
    "total_hospitalized": "Total_Hospitalized",
    "home_isolation": "Home_Isolation",
    "total_positive": "Total_Active_Cases",
    "positive_variation": "Change_in_Active_Cases",
    "new_positive_cases": "New_Positive_Cases",
    "recovered": "Discharged_Recovered",
    "deaths": "Deaths",
    "suspected_diagnosis_cases": "Cases_from_Diagnostic_Suspicion",
    "screening_cases": "Cases_from_Screening",
    "total_cases": "Total_Cases",
    "total_tests": "Tests_Swabs_Performed",
    "tested_cases": "People_Tested",
    "intensive_care_admissions": "ICU_Admissions",
    "molecular_positive_cases": "Total_Positives_Molecular_PCR",
    "rapid_antigen_positive_cases": "Total_Positives_Rapid_Antigen",
    "molecular_tests": "Molecular_PCR_Tests_Performed",
    "rapid_antigen_tests": "Rapid_Antigen_Tests_Performed",
}


REGIONAL_COLUMN_MAPPING = {
    "symptomatic_hospitalized": "Hospitalized_with_symptoms",
    "intensive_care": "Intensive_care_ICU",
    "total_hospitalized": "Total_hospitalized",
    "home_isolation": "Home_isolation",
    "total_positive": "Total_currently_positive",
    "positive_variation": "Change_in_total_positives",
    "new_positive_cases": "New_positives",
    "recovered": "Discharged_recovered",
    "deaths": "Deaths",
    "suspected_diagnosis_cases": "Cases_from_diagnostic_suspicion",
    "screening_cases": "Cases_from_screening",
    "total_cases": "Total_cases",
    "total_tests": "Swab_tests",
    "tested_cases": "People_tested",
    "intensive_care_admissions": "ICU_admissions_daily",
    "molecular_positive_cases": "Positives_by_PCR",
    "rapid_antigen_positive_cases": "Positives_by_rapid_antigen",
    "molecular_tests": "Molecular_PCR_tests",
    "rapid_antigen_tests": "Rapid_antigen_tests",
}


# ============================================================
# 5. DATABASE TARGET TABLES
# ============================================================

TARGET_TABLES = {
    "DimLocation": "DimLocation",
    "DimDate": "DimDate",
    "FactPopulation": "FactPopulation",
    "FactNational": "FactNational_metrics",
    "FactRegional": "FactRegional_Metrics",
    "ReportNotes": "ReportNotes",
}


# ============================================================
# 6. GENERAL VALIDATION FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def validate_columns(data):
    print_section("COLUMN VALIDATION")

    for table_name, expected in EXPECTED_COLUMNS.items():

        df = data[table_name]

        missing_columns = [
            col for col in expected
            if col not in df.columns
        ]

        extra_columns = [
            col for col in df.columns
            if col not in expected
        ]

        if missing_columns:
            raise ValueError(
                f"{table_name}: Missing columns: {missing_columns}"
            )

        if extra_columns:
            raise ValueError(
                f"{table_name}: Unexpected columns: {extra_columns}"
            )

        print(f"{table_name}: Columns validated successfully.")


def validate_keys(data):
    print_section("KEY VALIDATION")

    location = data["DimLocation"]
    dates = data["DimDate"]
    population = data["FactPopulation"]
    national = data["FactNational"]
    regional = data["FactRegional"]

    if location["Region_Code"].duplicated().any():
        raise ValueError("Duplicate Region_Code in DimLocation.")

    if dates["Date"].duplicated().any():
        raise ValueError("Duplicate Date in DimDate.")

    if population.duplicated(
        subset=["Region_Code", "Age_Range"]
    ).any():
        raise ValueError(
            "Duplicate Region_Code and Age_Range in FactPopulation."
        )

    if national["Date_Key"].duplicated().any():
        raise ValueError("Duplicate Date_Key in FactNational.")

    if regional.duplicated(
        subset=["Region_Code", "Date_Key"]
    ).any():
        raise ValueError(
            "Duplicate Region_Code and Date_Key in FactRegional."
        )

    print("All natural key checks passed.")


def validate_foreign_keys(data):
    print_section("FOREIGN KEY VALIDATION")

    valid_regions = set(
        data["DimLocation"]["Region_Code"].dropna()
    )

    valid_dates = set(
        pd.to_datetime(
            data["DimDate"]["Date"]
        ).dt.strftime("%Y%m%d")
    )

    population_regions = set(
        data["FactPopulation"]["Region_Code"].dropna()
    )

    regional_regions = set(
        data["FactRegional"]["Region_Code"].dropna()
    )

    national_dates = set(
        data["FactNational"]["Date_Key"].astype(str)
    )

    regional_dates = set(
        data["FactRegional"]["Date_Key"].astype(str)
    )

    if not population_regions.issubset(valid_regions):
        raise ValueError(
            "FactPopulation contains unknown Region_Code values."
        )

    if not regional_regions.issubset(valid_regions):
        raise ValueError(
            "FactRegional contains unknown Region_Code values."
        )

    if not national_dates.issubset(valid_dates):
        raise ValueError(
            "FactNational contains Date_Key values absent from DimDate."
        )

    if not regional_dates.issubset(valid_dates):
        raise ValueError(
            "FactRegional contains Date_Key values absent from DimDate."
        )

    print("All foreign key source references are valid.")


# ============================================================
# 7. LOAD ETL FILES
# ============================================================

def load_etl_files():

    print_section("READING ETL OUTPUT FILES")

    data = {}

    for table_name, file_path in FILE_PATHS.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        data[table_name] = df

        print(
            f"{table_name}: "
            f"{df.shape[0]:,} rows × {df.shape[1]} columns"
        )

    return data


# ============================================================
# 8. PREPARE DATA
# ============================================================

def normalize_date_column(df, column_name):

    df = df.copy()

    df[column_name] = pd.to_datetime(
        df[column_name],
        errors="coerce"
    ).dt.normalize()

    if df[column_name].isna().any():
        raise ValueError(
            f"Invalid or missing dates found in {column_name}."
        )

    return df


def convert_date_key(df):

    df = df.copy()

    df["Date_Key"] = (
        df["Date_Key"]
        .astype(str)
        .str.strip()
    )

    df["Date"] = pd.to_datetime(
        df["Date_Key"],
        format="%Y%m%d",
        errors="coerce"
    )

    if df["Date"].isna().any():
        raise ValueError(
            "Invalid Date_Key values found."
        )

    df["Date"] = df["Date"].dt.normalize()

    df.drop(columns=["Date_Key"], inplace=True)

    return df


def prepare_fact_data(data):

    print_section("PREPARING FACT DATA")

    prepared = {
        name: df.copy()
        for name, df in data.items()
    }

    # DimDate
    prepared["DimDate"] = normalize_date_column(
        prepared["DimDate"],
        "Date"
    )

    # FactNational
    national = convert_date_key(
        prepared["FactNational"]
    )

    national.rename(
        columns=NATIONAL_COLUMN_MAPPING,
        inplace=True
    )

    prepared["FactNational"] = national

    # FactRegional
    regional = convert_date_key(
        prepared["FactRegional"]
    )

    regional.rename(
        columns=REGIONAL_COLUMN_MAPPING,
        inplace=True
    )

    prepared["FactRegional"] = regional

    # ReportNotes
    prepared["ReportNotes"] = normalize_date_column(
        prepared["ReportNotes"],
        "Date"
    )

    print("Date conversion completed.")
    print("National column mapping completed.")
    print("Regional column mapping completed.")

    print("\nPrepared data shapes:")

    for name, df in prepared.items():
        print(f"{name}: {df.shape}")

    return prepared


# ============================================================
# 9. DATABASE ENGINE
# ============================================================

def get_database_engine():

    # Import only when a real database load is requested.
    # This allows local ETL preparation without pyodbc.

    if str(DATABASE_DIR) not in sys.path:
        sys.path.insert(0, str(DATABASE_DIR))

    from connection import engine

    return engine


# ============================================================
# 10. DATABASE SCHEMA VALIDATION
# ============================================================

def validate_sql_table_schema(
    connection,
    table_name,
    dataframe
):

    query = text("""
        SELECT
            COLUMN_NAME,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            COLUMNPROPERTY(
                OBJECT_ID(
                    QUOTENAME(TABLE_SCHEMA)
                    + '.'
                    + QUOTENAME(TABLE_NAME)
                ),
                COLUMN_NAME,
                'IsIdentity'
            ) AS IS_IDENTITY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema_name
          AND TABLE_NAME = :table_name
    """)

    result = connection.execute(
        query,
        {
            "schema_name": SCHEMA_NAME,
            "table_name": table_name,
        }
    )

    columns = result.mappings().all()

    if not columns:
        raise ValueError(
            f"SQL table dbo.{table_name} does not exist."
        )

    sql_columns = {
        row["COLUMN_NAME"]: row
        for row in columns
    }

    dataframe_columns = set(dataframe.columns)

    missing_sql_columns = [
        col for col in dataframe_columns
        if col not in sql_columns
    ]

    if missing_sql_columns:
        raise ValueError(
            f"dbo.{table_name}: These CSV columns do not exist "
            f"in SQL Server: {missing_sql_columns}"
        )

    # Check that every required SQL column receives a value.
    for column_name, metadata in sql_columns.items():

        if column_name in dataframe_columns:

            if (
                metadata["IS_NULLABLE"] == "NO"
                and dataframe[column_name].isna().any()
            ):
                raise ValueError(
                    f"dbo.{table_name}.{column_name} is NOT NULL "
                    "but the dataframe contains missing values."
                )

        else:
            is_identity = metadata["IS_IDENTITY"] == 1
            has_default = metadata["COLUMN_DEFAULT"] is not None

            if (
                metadata["IS_NULLABLE"] == "NO"
                and not is_identity
                and not has_default
            ):
                raise ValueError(
                    f"dbo.{table_name} has a required column "
                    f"'{column_name}' that is not supplied by the CSV."
                )

    print(f"dbo.{table_name}: SQL schema validated.")



def check_database_readiness(connection, data):

    print_section("DATABASE READINESS CHECK")

    database_name = connection.execute(
        text("SELECT DB_NAME()")
    ).scalar()

    if database_name != DATABASE_NAME:
        raise ValueError(
            f"Connected to '{database_name}', "
            f"expected '{DATABASE_NAME}'."
        )

    print(f"Connected database: {database_name}")

    # These are the columns that will ACTUALLY be inserted
    # after surrogate key mapping.

    sql_input_columns = {
        "DimLocation": [
            "Region_Code",
            "Region_Name",
            "Region_Abbreviation",
            "Latitude",
            "Longitude",
            "NUTS_1_Code",
            "NUTS_1_Description",
            "NUTS_2_Code",
        ],

        "DimDate": [
            "Date",
        ],

        "FactPopulation": [
            "location_id",
            "Age_Range",
            "Total_Male",
            "Total_Female",
            "Total_Population",
        ],

        "FactNational": [
            "date_id",
            *NATIONAL_COLUMN_MAPPING.values()
        ],

        "FactRegional": [
            "date_id",
            "location_id",
            *REGIONAL_COLUMN_MAPPING.values()
        ],

        "ReportNotes": [
            "date_id",
            "location_id",
            "source_type",
            "Note",
        ],
    }

    for data_name, table_name in TARGET_TABLES.items():

        # Validate the final SQL columns, not the original CSV columns.
        schema_check_df = pd.DataFrame(
            columns=sql_input_columns[data_name]
        )

        validate_sql_table_schema(
            connection,
            table_name,
            schema_check_df
        )

        row_count = connection.execute(
            text(
                f"SELECT COUNT(*) FROM "
                f"[{SCHEMA_NAME}].[{table_name}]"
            )
        ).scalar_one()

        if row_count != 0:
            raise ValueError(
                f"dbo.{table_name} is not empty "
                f"({row_count:,} existing rows). "
                "Loading stopped to prevent duplicate insertion."
            )

        print(
            f"dbo.{table_name}: empty and ready."
        )

    print("\nDatabase readiness checks passed.")



# ============================================================
# 11. DIMENSION LOADING AND SURROGATE KEY LOOKUPS
# ============================================================

def load_dimensions_and_get_lookups(data, connection):

    print_section("LOADING DIMENSION TABLES")

    location_df = data["DimLocation"].copy()
    date_df = data["DimDate"].copy()

    location_df.to_sql(
        name="DimLocation",
        con=connection,
        schema=SCHEMA_NAME,
        if_exists="append",
        index=False,
        chunksize=CHUNK_SIZE
    )

    print(
        f"Inserted {len(location_df):,} rows into DimLocation."
    )

    date_df.to_sql(
        name="DimDate",
        con=connection,
        schema=SCHEMA_NAME,
        if_exists="append",
        index=False,
        chunksize=CHUNK_SIZE
    )

    print(
        f"Inserted {len(date_df):,} rows into DimDate."
    )

    # Retrieve generated identity keys.
    location_lookup = pd.read_sql(
        text("""
            SELECT location_id, Region_Code
            FROM dbo.DimLocation
        """),
        connection
    )

    date_lookup = pd.read_sql(
        text("""
            SELECT date_id, [Date]
            FROM dbo.DimDate
        """),
        connection
    )

    location_lookup["Region_Code"] = pd.to_numeric(
        location_lookup["Region_Code"],
        errors="raise"
    ).astype(int)

    date_lookup["Date"] = pd.to_datetime(
        date_lookup["Date"]
    ).dt.normalize()

    if location_lookup["Region_Code"].duplicated().any():
        raise ValueError(
            "Duplicate Region_Code found in SQL DimLocation."
        )

    if date_lookup["Date"].duplicated().any():
        raise ValueError(
            "Duplicate Date found in SQL DimDate."
        )

    if len(location_lookup) != len(location_df):
        raise ValueError(
            "DimLocation lookup count does not match inserted rows."
        )

    if len(date_lookup) != len(date_df):
        raise ValueError(
            "DimDate lookup count does not match inserted rows."
        )

    print("Surrogate key lookups retrieved successfully.")

    return location_lookup, date_lookup


# ============================================================
# 12. FACT POPULATION
# ============================================================

def load_fact_population(
    data,
    location_lookup,
    connection
):

    print_section("LOADING FACT POPULATION")

    df = data["FactPopulation"].copy()

    df = df.merge(
        location_lookup,
        on="Region_Code",
        how="left",
        validate="many_to_one"
    )

    if df["location_id"].isna().any():
        raise ValueError(
            "Some FactPopulation records could not be mapped "
            "to location_id."
        )

    df.drop(
        columns=["Region_Code"],
        inplace=True
    )

    df = df[
        [
            "location_id",
            "Age_Range",
            "Total_Male",
            "Total_Female",
            "Total_Population",
        ]
    ]

    df.to_sql(
        name="FactPopulation",
        con=connection,
        schema=SCHEMA_NAME,
        if_exists="append",
        index=False,
        chunksize=CHUNK_SIZE
    )

    print(
        f"Inserted {len(df):,} rows into FactPopulation."
    )


# ============================================================
# 13. FACT NATIONAL METRICS
# ============================================================

def load_fact_national(
    data,
    date_lookup,
    connection
):

    print_section("LOADING FACT NATIONAL METRICS")

    df = data["FactNational"].copy()

    df = df.merge(
        date_lookup,
        on="Date",
        how="left",
        validate="many_to_one"
    )

    if df["date_id"].isna().any():
        raise ValueError(
            "Some FactNational records could not be mapped "
            "to date_id."
        )

    sql_measure_columns = list(
        NATIONAL_COLUMN_MAPPING.values()
    )

    df = df[
        [
            "date_id",
            *sql_measure_columns
        ]
    ]

    df.to_sql(
        name="FactNational_metrics",
        con=connection,
        schema=SCHEMA_NAME,
        if_exists="append",
        index=False,
        chunksize=CHUNK_SIZE
    )

    print(
        f"Inserted {len(df):,} rows into FactNational_metrics."
    )


# ============================================================
# 14. FACT REGIONAL METRICS
# ============================================================

def load_fact_regional(
    data,
    location_lookup,
    date_lookup,
    connection
):

    print_section("LOADING FACT REGIONAL METRICS")

    df = data["FactRegional"].copy()

    df = df.merge(
        location_lookup,
        on="Region_Code",
        how="left",
        validate="many_to_one"
    )

    df = df.merge(
        date_lookup,
        on="Date",
        how="left",
        validate="many_to_one"
    )

    if df["location_id"].isna().any():
        raise ValueError(
            "Some FactRegional records could not be mapped "
            "to location_id."
        )

    if df["date_id"].isna().any():
        raise ValueError(
            "Some FactRegional records could not be mapped "
            "to date_id."
        )

    sql_measure_columns = list(
        REGIONAL_COLUMN_MAPPING.values()
    )

    df = df[
        [
            "date_id",
            "location_id",
            *sql_measure_columns
        ]
    ]

    df.to_sql(
        name="FactRegional_Metrics",
        con=connection,
        schema=SCHEMA_NAME,
        if_exists="append",
        index=False,
        chunksize=CHUNK_SIZE
    )

    print(
        f"Inserted {len(df):,} rows into FactRegional_Metrics."
    )


# ============================================================
# 15. REPORT NOTES
# ============================================================

def load_report_notes(
    data,
    date_lookup,
    location_lookup,
    connection
):

    print_section("LOADING REPORT NOTES")

    df = data["ReportNotes"].copy()

    # Map Date to date_id.
    df = df.merge(
        date_lookup,
        on="Date",
        how="left",
        validate="many_to_one"
    )

    if df["date_id"].isna().any():
        raise ValueError(
            "Some ReportNotes records could not be mapped "
            "to date_id."
        )

    # Map Region_Code to location_id.
    # Null Region_Code is allowed because location_id is nullable.
    region_to_location = (
        location_lookup
        .set_index("Region_Code")["location_id"]
    )

    df["location_id"] = df["Region_Code"].map(
        region_to_location
    )

    # Check that every non-null Region_Code was mapped.
    invalid_regions = (
        df["Region_Code"].notna()
        & df["location_id"].isna()
    )

    if invalid_regions.any():
        raise ValueError(
            "Some non-null ReportNotes Region_Code values "
            "could not be mapped to location_id."
        )

    df = df[
        [
            "date_id",
            "location_id",
            "source_type",
            "Note",
        ]
    ]

    # Convert missing location IDs to actual None values.
    df["location_id"] = (
        df["location_id"]
        .astype(object)
        .where(df["location_id"].notna(), None)
    )

    if df["Note"].isna().any():
        raise ValueError(
            "ReportNotes contains NULL Note values."
        )

    df.to_sql(
        name="ReportNotes",
        con=connection,
        schema=SCHEMA_NAME,
        if_exists="append",
        index=False,
        chunksize=CHUNK_SIZE
    )

    print(
        f"Inserted {len(df):,} rows into ReportNotes."
    )


# ============================================================
# 16. POST-LOAD VALIDATION
# ============================================================

def validate_loaded_row_counts(data, connection):

    print_section("POST-LOAD VALIDATION")

    expected_counts = {
        "DimLocation": len(data["DimLocation"]),
        "DimDate": len(data["DimDate"]),
        "FactPopulation": len(data["FactPopulation"]),
        "FactNational_metrics": len(data["FactNational"]),
        "FactRegional_Metrics": len(data["FactRegional"]),
        "ReportNotes": len(data["ReportNotes"]),
    }

    total_loaded = 0

    for table_name, expected_count in expected_counts.items():

        actual_count = connection.execute(
            text(
                f"SELECT COUNT(*) "
                f"FROM [{SCHEMA_NAME}].[{table_name}]"
            )
        ).scalar_one()

        print(
            f"{table_name}: "
            f"Expected = {expected_count:,}, "
            f"Actual = {actual_count:,}"
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Row count mismatch in dbo.{table_name}."
            )

        total_loaded += actual_count

    print("\nAll table row counts match.")

    print(
        f"Total inserted rows: {total_loaded:,}"
    )


# ============================================================
# 17. EXECUTE FINAL DATABASE LOAD
# ============================================================

def execute_final_load(data):

    print_section("STARTING FINAL DATABASE LOAD")

    engine = get_database_engine()

    try:

        # A single transaction covers:
        # readiness checks, dimensions, facts, notes,
        # and post-load validation.

        with engine.begin() as connection:

            check_database_readiness(
                connection,
                data
            )

            location_lookup, date_lookup = (
                load_dimensions_and_get_lookups(
                    data,
                    connection
                )
            )

            load_fact_population(
                data,
                location_lookup,
                connection
            )

            load_fact_national(
                data,
                date_lookup,
                connection
            )

            load_fact_regional(
                data,
                location_lookup,
                date_lookup,
                connection
            )

            load_report_notes(
                data,
                date_lookup,
                location_lookup,
                connection
            )

            validate_loaded_row_counts(
                data,
                connection
            )

        print_section("DATABASE LOAD SUCCESSFUL")

        print(
            "All six tables were loaded successfully."
        )

        print(
            "The transaction was committed."
        )

    except Exception as error:

        print_section("DATABASE LOAD FAILED")

        print(f"Error: {error}")

        print(
            "The transaction was rolled back. "
            "No partial load should remain."
        )

        raise


# ============================================================
# 18. MAIN EXECUTION
# ============================================================

def main():

    print_section("EPISHIELD AI - FINAL ETL LOAD")

    # Step 1: Read the six prepared ETL files.
    data = load_etl_files()

    # Step 2: Validate input structure.
    validate_columns(data)

    # Step 3: Validate natural keys.
    validate_keys(data)

    # Step 4: Validate source relationships.
    validate_foreign_keys(data)

    # Step 5: Prepare dates and SQL measure names.
    prepared_data = prepare_fact_data(data)

    print_section("ETL PREPARATION COMPLETED")

    print(
        "All local ETL preparation checks passed."
    )

    print(
        "No database connection or insertion has been performed."
    )

    if RUN_DATABASE_LOAD:

        execute_final_load(prepared_data)

    else:

        print_section("DATABASE LOAD SKIPPED")

        print(
            f"RUN_DATABASE_LOAD = {RUN_DATABASE_LOAD}"
        )

        print(
            "Database loading is disabled."
        )

if __name__ == "__main__":
    main()