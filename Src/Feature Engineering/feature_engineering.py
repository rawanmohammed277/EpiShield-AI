from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 0) PATHS AND CONFIGURATION
# ============================================================

def find_project_root() -> Path:
    """Find the repository root by locating Src/Processed Data."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "Src" / "Processed Data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the project root. Expected a folder named "
        "'Src/Processed Data' in the current directory or one of its parents."
    )


PROJECT_DIR = find_project_root()
DATA_DIR = PROJECT_DIR / "Src" / "Processed Data"
INPUT_FILE = DATA_DIR / "regional_clean.csv"

OUTPUT_DIR = (
    PROJECT_DIR
    / "Src"
    / "Feature Engineering"
    / "outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cleaning's approved target is next-day, so Feature Engineering uses HORIZON=1.
HORIZON = 1

# Conservative default: enable these only if the values are available
# at the actual time the t+1 prediction is generated.
USE_HOSPITAL_FEATURES = False

RANDOM_STATE = 42
SAMPLE_SIZE = 3000

report = []


def log(message: str) -> None:
    print(message)
    report.append(str(message))


# ============================================================
# 1) LOAD AND VALIDATE CLEANED DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Required input file was not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Normalize column names only for matching; preserve original columns in source.
required_columns = [
    "date",
    "codice_regione",
    "denominazione_regione",
    "nuovi_positivi",
]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(
        "regional_clean.csv is missing required columns: "
        f"{missing}\nAvailable columns: {df.columns.tolist()}"
    )

# Parse dates and identifiers.
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["codice_regione"] = df["codice_regione"].astype("string").str.strip()
df["denominazione_regione"] = (
    df["denominazione_regione"].astype("string").str.strip()
)
df["nuovi_positivi"] = pd.to_numeric(df["nuovi_positivi"], errors="coerce")

# Optional hospital columns: retain only if present and explicitly enabled.
optional_hospital_columns = ["totale_ospedalizzati", "terapia_intensiva"]
available_hospital_columns = [
    col for col in optional_hospital_columns if col in df.columns
]

selected_columns = required_columns.copy()
selected_columns.remove("nuovi_positivi")
selected_columns.append("nuovi_positivi")
selected_columns += available_hospital_columns

df = df[selected_columns].copy()
df = df.rename(
    columns={
        "nuovi_positivi": "source_new_cases",
        "totale_ospedalizzati": "hospitalized",
        "terapia_intensiva": "icu",
    }
)

for col in ["hospitalized", "icu"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows without essential identifiers or target source value.
before_rows = len(df)
df = df.dropna(
    subset=["date", "codice_regione", "denominazione_regione", "source_new_cases"]
).copy()
dropped_invalid_rows = before_rows - len(df)

# Keep region codes as strings, including leading zeros.
df["codice_regione"] = df["codice_regione"].astype(str)

# Check and report duplicate region/date keys; do not silently aggregate them.
duplicate_count = int(df.duplicated(["codice_regione", "date"]).sum())
if duplicate_count:
    raise ValueError(
        f"Found {duplicate_count} duplicate (codice_regione, date) rows. "
        "Resolve duplicates before Feature Engineering; automatic aggregation "
        "could hide source-data issues."
    )

df = df.sort_values(["codice_regione", "date"]).reset_index(drop=True)

# Preserve negative source corrections. No clip(lower=0) is applied.
df["new_cases"] = df["source_new_cases"].astype(float)
df["is_negative_source_correction"] = (df["new_cases"] < 0).astype("int8")

log("=== Data Preparation ===")
log(f"Input file: {INPUT_FILE}")
log(f"Output directory: {OUTPUT_DIR}")
log(f"Rows after removing invalid identifiers/target: {len(df)}")
log(f"Rows removed for missing essential values: {dropped_invalid_rows}")
log(f"Number of regions: {df['codice_regione'].nunique()}")
log(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
log(f"Duplicate region/date keys: {duplicate_count}")

# Check missing calendar days within each region.
date_gaps = (
    df.groupby("codice_regione")["date"]
    .diff()
    .dropna()
    .ne(pd.Timedelta(days=1))
)
log(f"Non-consecutive date transitions within regions: {int(date_gaps.sum())}")
log(
    "Negative source corrections preserved (not clipped): "
    f"{int(df['is_negative_source_correction'].sum())}"
)

# Hospital features are opt-in only after operational availability is confirmed.
hospital_features_enabled = (
    USE_HOSPITAL_FEATURES
    and "hospitalized" in df.columns
    and "icu" in df.columns
)
if USE_HOSPITAL_FEATURES and not hospital_features_enabled:
    log(
        "WARNING: Hospital features were requested, but one or both source "
        "columns are unavailable. They will be excluded."
    )
elif hospital_features_enabled:
    log(
        "Hospitalized/ICU features ENABLED. Confirm that both are available "
        "at the real prediction cutoff."
    )
else:
    log(
        "Hospitalized/ICU features DISABLED by default pending confirmation "
        "of operational availability."
    )


# ============================================================
# 2) FEATURE ENGINEERING
# ============================================================

def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features using information available on date t only."""
    data = data.sort_values(
        ["codice_regione", "date"]
    ).reset_index(drop=True).copy()

    grouped_cases = data.groupby("codice_regione", sort=False)["new_cases"]

    # Calendar features.
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek  # Monday=0, Sunday=6
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    data["year"] = data["date"].dt.year
    data["is_weekend"] = (data["day_of_week"] >= 5).astype("int8")

    # Current day and lag features.
    data["cases_today"] = data["new_cases"]
    for lag in [1, 2, 3, 7, 14]:
        data[f"cases_lag_{lag}"] = grouped_cases.shift(lag)

    # Rolling features include day t and past days only.
    for window in [7, 14, 30]:
        data[f"rolling_mean_{window}"] = grouped_cases.transform(
            lambda series: series.rolling(window=window, min_periods=window).mean()
        )

    data["rolling_std_7"] = grouped_cases.transform(
        lambda series: series.rolling(window=7, min_periods=7).std()
    )
    data["rolling_max_7"] = grouped_cases.transform(
        lambda series: series.rolling(window=7, min_periods=7).max()
    )

    # Additional trend features.
    data["daily_change"] = data["cases_today"] - data["cases_lag_1"]

    previous_7_day_mean = grouped_cases.transform(
        lambda series: series.shift(7).rolling(window=7, min_periods=7).mean()
    )
    data["growth_rate_7"] = (
        data["rolling_mean_7"] - previous_7_day_mean
    ) / (previous_7_day_mean.abs() + 1.0)

    # Negative correction indicator for the current day.
    data["is_negative_source_correction"] = (
        data["new_cases"] < 0
    ).astype("int8")

    # Optional hospital indicators: only enabled after cutoff availability check.
    if hospital_features_enabled:
        data["hospitalized"] = data["hospitalized"]
        data["icu"] = data["icu"]

    return data


full = make_features(df)

# One-day-ahead target. Require the next observation to be exactly the next day.
full["target_date"] = full.groupby("codice_regione")["date"].shift(-HORIZON)
full["target_next_day_new_cases"] = (
    full.groupby("codice_regione")["new_cases"].shift(-HORIZON)
)

consecutive_target_day = (
    full["target_date"] - full["date"]
).eq(pd.Timedelta(days=HORIZON))
full.loc[~consecutive_target_day, "target_next_day_new_cases"] = np.nan
full.loc[~consecutive_target_day, "target_date"] = pd.NaT

# Define the model feature list explicitly. Target columns are never predictors.
FEATURES = [
    "day",
    "day_of_week",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "cases_today",
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_7",
    "cases_lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_max_7",
    "daily_change",
    "growth_rate_7",
    "is_negative_source_correction",
]

if hospital_features_enabled:
    FEATURES += ["hospitalized", "icu"]

# Explicitly guard against accidentally including target or future-date columns.
for forbidden in ["target_next_day_new_cases", "target_date"]:
    if forbidden in FEATURES:
        raise AssertionError(f"Leakage detected: {forbidden} is in FEATURES.")


# ============================================================
# 3) FEATURE AND LEAKAGE VALIDATION
# ============================================================

log("\n=== Feature Validation ===")

missing_feature_rows = int(
    full[FEATURES + ["target_next_day_new_cases"]].isna().any(axis=1).sum()
)
log(
    "Rows with expected missing values from lags/rolling/target: "
    f"{missing_feature_rows}"
)

# Verify lag and target values against the original time series.
series_by_region_date = full.set_index(
    ["codice_regione", "date"]
)["new_cases"]


def values_at_offset(sample_rows: pd.DataFrame, offset_days: int) -> np.ndarray:
    index = pd.MultiIndex.from_arrays(
        [
            sample_rows["codice_regione"].to_numpy(),
            (sample_rows["date"] + pd.Timedelta(days=offset_days)).to_numpy(),
        ],
        names=["codice_regione", "date"],
    )
    return series_by_region_date.reindex(index).to_numpy(dtype=float)


valid_for_checks = full.dropna(
    subset=FEATURES + ["target_next_day_new_cases"]
)
if len(valid_for_checks):
    sample = valid_for_checks.sample(
        n=min(SAMPLE_SIZE, len(valid_for_checks)),
        random_state=RANDOM_STATE,
    )

    lag_7_ok = np.allclose(
        sample["cases_lag_7"].to_numpy(dtype=float),
        values_at_offset(sample, -7),
        equal_nan=True,
    )
    lag_14_ok = np.allclose(
        sample["cases_lag_14"].to_numpy(dtype=float),
        values_at_offset(sample, -14),
        equal_nan=True,
    )
    target_ok = np.allclose(
        sample["target_next_day_new_cases"].to_numpy(dtype=float),
        values_at_offset(sample, HORIZON),
        equal_nan=True,
    )

    manual_rolling_7 = np.mean(
        [values_at_offset(sample, -day) for day in range(0, 7)],
        axis=0,
    )
    rolling_7_ok = np.allclose(
        sample["rolling_mean_7"].to_numpy(dtype=float),
        manual_rolling_7,
        equal_nan=True,
    )
else:
    lag_7_ok = lag_14_ok = target_ok = rolling_7_ok = False

log(f"cases_lag_7 matches the value 7 days earlier: {lag_7_ok}")
log(f"cases_lag_14 matches the value 14 days earlier: {lag_14_ok}")
log(f"Next-day target matches t+1 source value: {target_ok}")
log(f"rolling_mean_7 matches manual calculation: {rolling_7_ok}")

# Cutoff test: features for dates <= cutoff must not change when later rows exist.
cutoff = pd.Timestamp("2022-06-30")
cut_data = df[df["date"] <= cutoff].copy()
cut_features = make_features(cut_data)

compare_columns = ["codice_regione", "date"] + FEATURES
merged = cut_features[compare_columns].merge(
    full[compare_columns],
    on=["codice_regione", "date"],
    how="inner",
    suffixes=("_cut", "_full"),
)

cutoff_test_ok = True
for column in FEATURES:
    left = merged[f"{column}_cut"].to_numpy(dtype=float)
    right = merged[f"{column}_full"].to_numpy(dtype=float)
    if not np.allclose(left, right, equal_nan=True):
        cutoff_test_ok = False
        log(f"WARNING: Feature changed after cutoff for column: {column}")

log(
    "Historical feature cutoff test (no future data influence): "
    f"{cutoff_test_ok}"
)

# Check target dates and infinite values.
target_date_ok = (
    full.loc[full["target_next_day_new_cases"].notna(), "target_date"]
    - full.loc[full["target_next_day_new_cases"].notna(), "date"]
).eq(pd.Timedelta(days=1)).all()

numeric_values = full[FEATURES + ["target_next_day_new_cases"]].select_dtypes(
    include=[np.number]
)
infinite_count = int(
    np.isinf(numeric_values.to_numpy(dtype=float)).sum()
)

log(f"All populated targets point exactly one day ahead: {target_date_ok}")
log(f"Number of infinite numeric values: {infinite_count}")
log(f"Minimum observed source daily cases: {full['new_cases'].min()}")

# Negative targets are retained in the engineered audit dataset, but excluded
# from supervised count-model splits rather than being altered to zero.
negative_target_count = int(
    (full["target_next_day_new_cases"] < 0).sum()
)
log(
    "Negative next-day target rows retained in full output but excluded "
    f"from model splits: {negative_target_count}"
)


# ============================================================
# 4) FINAL DATASET AND TEMPORAL SPLIT
# ============================================================

ID_COLUMNS = [
    "date",
    "target_date",
    "codice_regione",
    "denominazione_regione",
]

final = full.dropna(
    subset=FEATURES + ["target_next_day_new_cases", "target_date"]
).copy()

# Count targets should be non-negative. Keep negative source corrections
# visible in the full engineered output, but exclude negative target rows
# from model training/evaluation rather than clipping them.
final_model = final[
    final["target_next_day_new_cases"] >= 0
].copy()

if final_model.empty:
    raise ValueError(
        "No valid rows remain for modeling after feature/target validation."
    )

# One-hot encode region IDs; fit columns on the complete modeling dataset.
region_dummies = pd.get_dummies(
    final_model["codice_regione"].astype(str),
    prefix="region",
    dtype="int8",
)

# Apply the same region columns to the full dataset.
full_region_dummies = pd.get_dummies(
    full["codice_regione"].astype(str),
    prefix="region",
    dtype="int8",
).reindex(
    columns=region_dummies.columns,
    fill_value=0,
)

full = pd.concat([full, full_region_dummies], axis=1)

final_model = pd.concat([final_model, region_dummies], axis=1)

X_COLUMNS = FEATURES + region_dummies.columns.tolist()
OUTPUT_COLUMNS = (
    ID_COLUMNS
    + X_COLUMNS
    + ["target_next_day_new_cases"]
)

final_model = final_model[OUTPUT_COLUMNS].copy()

# Chronological date split: 70% train, 15% validation, 15% test.
unique_dates = np.sort(final_model["date"].dropna().unique())
if len(unique_dates) < 10:
    raise ValueError(
        "Not enough unique dates for a reliable chronological split."
    )

split_70 = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.70)])
split_85 = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.85)])
gap = pd.Timedelta(days=HORIZON)

train = final_model[final_model["date"] <= split_70 - gap].copy()
validation = final_model[
    (final_model["date"] > split_70)
    & (final_model["date"] <= split_85 - gap)
].copy()
test = final_model[final_model["date"] > split_85].copy()

if train.empty or validation.empty or test.empty:
    raise ValueError(
        "At least one temporal split is empty. Review the date range and split logic."
    )

log("\n=== Temporal Split ===")
for split_name, split_data in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:
    log(
        f"{split_name}: {len(split_data)} rows | "
        f"{split_data['date'].min().date()} to "
        f"{split_data['date'].max().date()} | "
        f"{split_data['codice_regione'].nunique()} regions"
    )

no_date_overlap = (
    train["date"].max() < validation["date"].min()
    and validation["date"].max() < test["date"].min()
)
train_target_before_validation = (
    train["target_date"].max() < validation["date"].min()
)
validation_target_before_test = (
    validation["target_date"].max() < test["date"].min()
)

log(f"No date overlap between splits: {no_date_overlap}")
log(
    "Training targets end before validation begins: "
    f"{train_target_before_validation}"
)
log(
    "Validation targets end before test begins: "
    f"{validation_target_before_test}"
)
log(
    f"Final modeling dataset shape: {final_model.shape} | "
    f"Number of predictors: {len(X_COLUMNS)}"
)
log(
    "Missing values in final modeling dataset: "
    f"{int(final_model.isna().sum().sum())}"
)


# ============================================================
# 5) FEATURE DESCRIPTION
# ============================================================

feature_descriptions = [
    (
        "target_next_day_new_cases",
        "Target",
        "New cases on the following day (t+1) for the same region",
        "groupby(region).shift(-1), only when target_date is exactly t+1",
    ),
    ("day", "Time", "Day of month", "Extracted from date"),
    ("day_of_week", "Time", "Day of week; Monday=0 and Sunday=6", "Extracted from date"),
    ("month", "Time", "Calendar month", "Extracted from date"),
    ("quarter", "Time", "Calendar quarter", "Extracted from date"),
    ("year", "Time", "Calendar year", "Extracted from date"),
    ("is_weekend", "Time", "Whether the date is Saturday or Sunday", "day_of_week >= 5"),
    ("cases_today", "Current", "Reported new cases on prediction date t", "source_new_cases"),
    ("cases_lag_1", "Lag", "Reported new cases one day before t", "groupby(region).shift(1)"),
    ("cases_lag_2", "Lag", "Reported new cases two days before t", "groupby(region).shift(2)"),
    ("cases_lag_3", "Lag", "Reported new cases three days before t", "groupby(region).shift(3)"),
    ("cases_lag_7", "Lag", "Reported new cases seven days before t", "groupby(region).shift(7)"),
    ("cases_lag_14", "Lag", "Reported new cases fourteen days before t", "groupby(region).shift(14)"),
    ("rolling_mean_7", "Rolling", "Mean of the latest 7 days including t", "rolling(7).mean()"),
    ("rolling_mean_14", "Rolling", "Mean of the latest 14 days including t", "rolling(14).mean()"),
    ("rolling_mean_30", "Rolling", "Mean of the latest 30 days including t", "rolling(30).mean()"),
    ("rolling_std_7", "Rolling", "Standard deviation over the latest 7 days", "rolling(7).std()"),
    ("rolling_max_7", "Rolling", "Maximum over the latest 7 days", "rolling(7).max()"),
    ("daily_change", "Additional", "Difference between today's and yesterday's cases", "cases_today - cases_lag_1"),
    ("growth_rate_7", "Additional", "Relative change between two consecutive 7-day means", "(mean_7_t - mean_7_previous) / (abs(mean_7_previous) + 1)"),
    ("is_negative_source_correction", "Data quality", "Flag indicating a negative source daily value", "new_cases < 0"),
]

if hospital_features_enabled:
    feature_descriptions.extend(
        [
            ("hospitalized", "Additional", "Currently hospitalized patients", "totale_ospedalizzati; enabled only after cutoff availability confirmation"),
            ("icu", "Additional", "Patients in intensive care", "terapia_intensiva; enabled only after cutoff availability confirmation"),
        ]
    )

for region_column in region_dummies.columns:
    feature_descriptions.append(
        (
            region_column,
            "Region",
            f"One-hot indicator for region code {region_column.replace('region_', '')}",
            "pandas.get_dummies(codice_regione)",
        )
    )

description_df = pd.DataFrame(
    feature_descriptions,
    columns=["feature", "type", "meaning", "how_computed"],
)


# Ensure full dataset contains the same region columns used by the model.
full_region_dummies = pd.get_dummies(
    full["codice_regione"].astype(str),
    prefix="region",
    dtype="int8",
).reindex(
    columns=region_dummies.columns,
    fill_value=0,
)

# Remove old region columns if they already exist, then add the correct ones.
full = full.drop(
    columns=region_dummies.columns,
    errors="ignore",
)

full = pd.concat(
    [full, full_region_dummies],
    axis=1,
)


# ============================================================
# 6) SAVE OUTPUTS
# ============================================================

# Full engineered table includes negative next-day targets for auditability.
full.to_csv(
    OUTPUT_DIR / "feature_engineered_dataset.csv",
    index=False,
    encoding="utf-8-sig",
)

train.to_csv(
    OUTPUT_DIR / "train_dataset.csv",
    index=False,
    encoding="utf-8-sig",
)
validation.to_csv(
    OUTPUT_DIR / "validation_dataset.csv",
    index=False,
    encoding="utf-8-sig",
)
test.to_csv(
    OUTPUT_DIR / "test_dataset.csv",
    index=False,
    encoding="utf-8-sig",
)

description_df.to_csv(
    OUTPUT_DIR / "feature_description.csv",
    index=False,
    encoding="utf-8-sig",
)

# Final pass/fail summary.
checks = {
    "No duplicate region/date keys": duplicate_count == 0,
    "Lag 7 validation": lag_7_ok,
    "Lag 14 validation": lag_14_ok,
    "Next-day target validation": target_ok,
    "Rolling mean 7 validation": rolling_7_ok,
    "Historical cutoff leakage test": cutoff_test_ok,
    "Target date is exactly t+1": bool(target_date_ok),
    "No infinite values": infinite_count == 0,
    "No date overlap between splits": bool(no_date_overlap),
    "Train targets end before validation": bool(train_target_before_validation),
    "Validation targets end before test": bool(validation_target_before_test),
    "No missing values in final modeling dataset": (
        int(final_model.isna().sum().sum()) == 0
    ),
}

log("\n=== Final Validation Summary ===")
for check_name, passed in checks.items():
    log(f"{'PASS' if passed else 'FAIL'} | {check_name}")

validation_passed = all(checks.values())
log(f"\nOverall validation: {'PASS' if validation_passed else 'FAIL'}")

(OUTPUT_DIR / "feature_validation_report.txt").write_text(
    "\n".join(report),
    encoding="utf-8",
)

print("\nFeature Engineering finished.")
print(f"Outputs saved to: {OUTPUT_DIR}")
print(f"Overall validation: {'PASS' if validation_passed else 'FAIL'}")

if not validation_passed:
    raise RuntimeError(
        "One or more Feature Engineering validation checks failed. "
        "Review feature_validation_report.txt before using the datasets."
    )
