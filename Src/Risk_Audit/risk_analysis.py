
import pandas as pd
from pathlib import Path

# ==========================================
# 1. Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "Src"
    / "Data Engineering"
    / "Data_Cleaning"
    / "output"
    / "feature_engineered_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "Src" / "Risk_Audit" / "outputs"


# ==========================================
# 2. Check Dataset Path
# ==========================================

print("\n========== RISK AUDIT ==========")

print("\nChecking dataset path...")
print(f"Expected path: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATA_PATH}"
    )

print("Dataset found successfully!")


# ==========================================
# 3. Load Dataset
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("Dataset is empty!")

print("Dataset loaded successfully!")


# ==========================================
# 4. Dataset Overview
# ==========================================

print("\n========== DATASET OVERVIEW ==========")

print(f"\nDataset Shape: {df.shape}")

if "denominazione_regione" not in df.columns:
    raise ValueError("Region column not found!")

regions_count = df["denominazione_regione"].nunique()

print(f"Number of Regions: {regions_count}")

if "date" not in df.columns:
    raise ValueError("Date column not found!")

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print(
    f"Date Range: {df['date'].min()} "
    f"to {df['date'].max()}"
)


# ==========================================
# 5. Available Columns
# ==========================================

print("\n========== AVAILABLE COLUMNS ==========")

for column in df.columns:
    print(f"- {column}")


# ==========================================
# 6. Data Quality Audit
# ==========================================

print("\n========== DATA QUALITY AUDIT ==========")

missing_values = df.isnull().sum()

missing_report = pd.DataFrame({
    "Column": missing_values.index,
    "Missing_Count": missing_values.values,
    "Missing_Percentage": (
        missing_values.values / len(df) * 100
    ).round(2)
})

print("\nMissing Values:")

print(
    missing_report[
        missing_report["Missing_Count"] > 0
    ].to_string(index=False)
    if (missing_report["Missing_Count"] > 0).any()
    else "No missing values found!"
)

duplicates = df.duplicated(
    subset=["denominazione_regione", "date"]
)

print(
    "\nDuplicate Region-Date Records:",
    duplicates.sum()
)


# ==========================================
# 7. Regional Analysis
# ==========================================

print("\n========== REGIONAL ANALYSIS ==========")

metrics = [
    "cases_today",
    "cases_per_100k",
    "rolling_mean_7",
    "rolling_mean_7_per_100k",
    "growth_rate_7",
    "hospitalized",
    "hospitalized_per_100k",
    "icu",
    "icu_per_100k"
]

available_metrics = [
    col for col in metrics if col in df.columns
]

for col in available_metrics:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

if available_metrics:

    regional_analysis = (
        df.groupby("denominazione_regione")
        [available_metrics]
        .agg(["mean", "max"])
    )

    regional_analysis.columns = [
        f"{col}_{stat}"
        for col, stat in regional_analysis.columns
    ]

    regional_analysis = regional_analysis.reset_index()

    print(regional_analysis.to_string(index=False))

else:

    regional_analysis = pd.DataFrame()

    print("No regional metrics available.")


# ==========================================
# 8. Growth Analysis
# ==========================================

print("\n========== GROWTH ANALYSIS ==========")

if "growth_rate_7" in df.columns:

    growth_analysis = (
        df.groupby("denominazione_regione")
        ["growth_rate_7"]
        .agg(
            average_growth="mean",
            maximum_growth="max",
            minimum_growth="min"
        )
        .reset_index()
    )

    growth_analysis = growth_analysis.sort_values(
        "average_growth",
        ascending=False
    )

    print(growth_analysis.to_string(index=False))

else:

    growth_analysis = pd.DataFrame()

    print("Growth rate column not available.")


# ==========================================
# 9. Latest Regional Status
# ==========================================

print("\n========== LATEST REGIONAL STATUS ==========")

latest_status = (
    df.dropna(subset=["date"])
    .sort_values("date")
    .groupby("denominazione_regione")
    .tail(1)
    .copy()
)

status_columns = [
    "date",
    "denominazione_regione"
] + available_metrics

latest_status = latest_status[status_columns]

print(latest_status.to_string(index=False))


# ==========================================
# 10. Risk Score Calculation
# ==========================================

print("\n========== REGIONAL RISK SCORE ==========")

base_weights = {
    "cases_per_100k": 0.40,
    "rolling_mean_7_per_100k": 0.40,
    "growth_rate_7": 0.25,
    "hospitalized_per_100k": 0.20,
    "icu_per_100k": 0.15
}

selected_weights = {}

# Select cases indicator
if (
    "cases_per_100k" in latest_status.columns
    and latest_status["cases_per_100k"].notna().any()
):

    selected_weights["cases_per_100k"] = 0.40

elif (
    "rolling_mean_7_per_100k" in latest_status.columns
    and latest_status["rolling_mean_7_per_100k"].notna().any()
):

    selected_weights["rolling_mean_7_per_100k"] = 0.40


# Select other indicators
for metric in [
    "growth_rate_7",
    "hospitalized_per_100k",
    "icu_per_100k"
]:

    if (
        metric in latest_status.columns
        and latest_status[metric].notna().any()
    ):

        selected_weights[metric] = base_weights[metric]


if selected_weights:

    # Normalize weights
    total_weight = sum(selected_weights.values())

    normalized_weights = {
        metric: weight / total_weight
        for metric, weight in selected_weights.items()
    }

    print("\n========== INDICATOR WEIGHTS ==========")

    for metric, weight in normalized_weights.items():
        print(f"{metric}: {weight:.2%}")


    # Initialize risk score
    risk_scores = latest_status[
        ["date", "denominazione_regione"]
    ].copy()

    risk_scores["risk_score"] = 0.0


    # Initialize contribution report
    risk_contributions = latest_status[
        ["date", "denominazione_regione"]
    ].copy()


    # Calculate contributions
    for metric, weight in normalized_weights.items():

        values = pd.to_numeric(
            latest_status[metric],
            errors="coerce"
        )

        # Fill missing values with median
        values = values.fillna(values.median())

        # Calculate percentile score
        percentile_score = (
            values.rank(method="average", pct=True) * 100
        )

        # Calculate weighted contribution
        contribution = percentile_score * weight

        contribution_column = f"{metric}_contribution"

        risk_contributions[contribution_column] = (
            contribution.round(2)
        )

        # Add contribution to total risk score
        risk_scores["risk_score"] += contribution


    # Round final score
    risk_scores["risk_score"] = (
        risk_scores["risk_score"].round(2)
    )


    # Risk classification
    def classify_risk(score):

        if score >= 75:
            return "High"

        elif score >= 50:
            return "Moderate"

        else:
            return "Low"


    risk_scores["risk_level"] = (
        risk_scores["risk_score"].apply(classify_risk)
    )


    # Sort by risk score
    risk_scores = risk_scores.sort_values(
        "risk_score",
        ascending=False
    ).reset_index(drop=True)


    # Add ranking
    risk_scores.insert(
        0,
        "rank",
        range(1, len(risk_scores) + 1)
    )


    # Attach final score to contribution report
    score_mapping = risk_scores.set_index(
        "denominazione_regione"
    )["risk_score"]

    risk_contributions["risk_score"] = (
        risk_contributions["denominazione_regione"]
        .map(score_mapping)
    )


    risk_contributions = risk_contributions.sort_values(
        "risk_score",
        ascending=False
    ).reset_index(drop=True)


    print("\n========== REGIONAL RISK SCORES ==========")

    print(risk_scores.to_string(index=False))


    print("\n========== INDICATOR CONTRIBUTIONS ==========")

    print(risk_contributions.to_string(index=False))


else:

    risk_scores = pd.DataFrame()

    risk_contributions = pd.DataFrame()

    print("Not enough valid indicators to calculate risk score.")


# ==========================================
# 11. Risk Distribution
# ==========================================

print("\n========== RISK DISTRIBUTION ==========")

if not risk_scores.empty:

    risk_distribution = (
        risk_scores["risk_level"]
        .value_counts()
        .reindex(
            ["High", "Moderate", "Low"],
            fill_value=0
        )
        .rename_axis("Risk_Level")
        .reset_index(name="Number_of_Regions")
    )

    print(risk_distribution.to_string(index=False))

else:

    risk_distribution = pd.DataFrame()


# ==========================================
# 12. Save Reports
# ==========================================

print("\n========== SAVING REPORTS ==========")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

reports = {

    "dataset_overview.csv": pd.DataFrame({
        "Metric": [
            "Total Rows",
            "Total Columns",
            "Number of Regions",
            "Missing Values"
        ],

        "Value": [
            len(df),
            len(df.columns),
            df["denominazione_regione"].nunique(),
            int(df.isnull().sum().sum())
        ]
    }),

    "missing_values_audit.csv": missing_report,

    "regional_analysis.csv": regional_analysis,

    "regional_growth_analysis.csv": growth_analysis,

    "latest_regional_status.csv": latest_status,

    "regional_risk_scores.csv": risk_scores,

    "risk_indicator_contributions.csv": risk_contributions,

    "risk_distribution.csv": risk_distribution,

    "duplicate_records.csv": df.loc[duplicates].copy()
}


for filename, report in reports.items():

    report.to_csv(
        OUTPUT_DIR / filename,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Saved: {filename}")


# ==========================================
# 13. Final Verification
# ==========================================

print("\n========== FINAL VERIFICATION ==========")

print(f"Total Rows: {len(df)}")

print(f"Total Columns: {len(df.columns)}")

print(
    f"Total Missing Values: {df.isnull().sum().sum()}"
)

print(
    f"Total Duplicate Records: {duplicates.sum()}"
)

print(
    f"Total Regions: {df['denominazione_regione'].nunique()}"
)

if not risk_scores.empty:

    print(
        "\nHighest Relative Risk Region:",
        risk_scores.iloc[0]["denominazione_regione"]
    )

    print(
        "Risk Score:",
        risk_scores.iloc[0]["risk_score"]
    )

    print(
        "Risk Level:",
        risk_scores.iloc[0]["risk_level"]
    )

print("\nReports Location:")

print(OUTPUT_DIR)

print("\nRISK AUDIT COMPLETED SUCCESSFULLY!")
