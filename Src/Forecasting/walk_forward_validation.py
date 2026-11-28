
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# 1. PATHS AND SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "Src" / "Feature Engineering" / "outputs"

TRAIN_PATH = DATA_DIR / "train_dataset.csv"
VALIDATION_PATH = DATA_DIR / "validation_dataset.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "walk_forward_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_next_day_new_cases"
DATE_COL = "date"

N_FOLDS = 3


# =========================================================
# 2. LOAD DATA
# =========================================================

def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    for df in [train_df, validation_df]:

        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        df.dropna(subset=[TARGET], inplace=True)

        # Exclude negative target values
        df.drop(df[df[TARGET] < 0].index, inplace=True)

    train_df = train_df.sort_values(DATE_COL).reset_index(drop=True)
    validation_df = validation_df.sort_values(DATE_COL).reset_index(drop=True)

    print("DATA LOADED SUCCESSFULLY")
    print("-" * 50)

    print("Training rows:", len(train_df))
    print("Validation rows:", len(validation_df))

    print(
        "Training period:",
        train_df[DATE_COL].min().date(),
        "to",
        train_df[DATE_COL].max().date()
    )

    print(
        "Validation period:",
        validation_df[DATE_COL].min().date(),
        "to",
        validation_df[DATE_COL].max().date()
    )

    return train_df, validation_df


# =========================================================
# 3. ADD EXPERIMENTAL FEATURES
# =========================================================

def add_features(df):

    df = df.copy()

    today = df["cases_today"].clip(lower=0)
    mean7 = df["rolling_mean_7"].clip(lower=0)
    std7 = df["rolling_std_7"].clip(lower=0)
    lag1 = df["cases_lag_1"].clip(lower=0)

    safe_mean = mean7 + 1

    # Enhanced features
    df["cases_today_to_mean7_ratio"] = today / safe_mean

    df["cases_today_minus_mean7"] = today - mean7

    df["volatility_ratio"] = std7 / safe_mean

    df["lag1_to_mean7_ratio"] = lag1 / safe_mean

    # Spike features
    df["positive_daily_change"] = (today - lag1).clip(lower=0)

    df["spike_ratio"] = today / safe_mean

    return df


# =========================================================
# 4. DEFINE FOUR FEATURE VARIANTS
# =========================================================

VARIANTS = {
    "Baseline": [],

    "Enhanced": [
        "cases_today_to_mean7_ratio",
        "cases_today_minus_mean7",
        "volatility_ratio",
        "lag1_to_mean7_ratio"
    ],

    "Spike": [
        "positive_daily_change",
        "spike_ratio"
    ],

    "Enhanced_Spike": [
        "cases_today_to_mean7_ratio",
        "cases_today_minus_mean7",
        "volatility_ratio",
        "lag1_to_mean7_ratio",
        "positive_daily_change",
        "spike_ratio"
    ]
}


# =========================================================
# 5. PREPARE FEATURES
# =========================================================

EXCLUDED_COLUMNS = [
    TARGET,
    "date",
    "target_date",
    "codice_regione",
    "denominazione_regione"
]



def prepare_features(train_df, validation_df, extra_features):

    ALL_EXPERIMENTAL_FEATURES = [
        "cases_today_to_mean7_ratio",
        "cases_today_minus_mean7",
        "volatility_ratio",
        "lag1_to_mean7_ratio",
        "positive_daily_change",
        "spike_ratio"
    ]

    numeric_columns = train_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # Keep original features only
    features = [
        col for col in numeric_columns
        if col not in EXCLUDED_COLUMNS
        and col not in ALL_EXPERIMENTAL_FEATURES
    ]

    # Add only the features of the current variant
    features.extend(extra_features)

    X_train = train_df.reindex(columns=features).copy()

    X_validation = validation_df.reindex(columns=features).copy()

    # Learn missing-value replacements from training only
    medians = X_train.median()

    X_train = X_train.fillna(medians).fillna(0)

    X_validation = X_validation.fillna(medians).fillna(0)

    return X_train, X_validation, features



# =========================================================
# 6. CREATE WALK-FORWARD FOLDS
# =========================================================

def create_folds(validation_df):

    unique_dates = np.sort(
        validation_df[DATE_COL].unique()
    )

    date_blocks = np.array_split(unique_dates, N_FOLDS)

    folds = []

    for fold_number, block in enumerate(date_blocks, start=1):

        validation_start = pd.Timestamp(block[0])

        validation_end = pd.Timestamp(block[-1])

        # Keep a one-day gap before each validation fold
        training_end = validation_start - pd.Timedelta(days=2)

        folds.append({
            "fold": fold_number,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "training_end": training_end
        })

    return folds


# =========================================================
# 7. CREATE MODEL
# =========================================================

def make_model():

    return XGBRegressor(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )


# =========================================================
# 8. RUN WALK-FORWARD VALIDATION
# =========================================================



def main():

    train_df, validation_df = load_data()

    train_df = add_features(train_df)
    validation_df = add_features(validation_df)

    folds = create_folds(validation_df)

    print("\nFEATURE COUNTS")

    for variant_name, extra_features in VARIANTS.items():
        print(
            variant_name,
            "Expected features:",
            41 + len(extra_features)
        )

    all_predictions = []
    fold_results = []

    print("\nSTARTING WALK-FORWARD VALIDATION")

    for fold_info in folds:

        fold_number = fold_info["fold"]
        validation_start = fold_info["validation_start"]
        validation_end = fold_info["validation_end"]
        training_end = fold_info["training_end"]

        print("\n" + "=" * 60)
        print(f"FOLD {fold_number}")
        print("Training until:", training_end.date())
        print("Validation:", validation_start.date(), "to", validation_end.date())

        # Expanding training window:
        # Original training data + previous validation dates
        previous_validation = validation_df[
            validation_df[DATE_COL] <= training_end
        ]

        fold_train = pd.concat(
            [train_df, previous_validation],
            ignore_index=True
        )

        fold_validation = validation_df[
            (validation_df[DATE_COL] >= validation_start) &
            (validation_df[DATE_COL] <= validation_end)
        ].copy()

        print("Fold training rows:", len(fold_train))
        print("Fold validation rows:", len(fold_validation))

        for variant_name, extra_features in VARIANTS.items():

            X_train, X_val, features = prepare_features(
                fold_train,
                fold_validation,
                extra_features
            )

            y_train = fold_train[TARGET].clip(lower=0)
            y_val = fold_validation[TARGET].clip(lower=0)

            model = make_model()

            # Train using log1p transformation
            model.fit(X_train, np.log1p(y_train))

            predictions = np.expm1(model.predict(X_val))
            predictions = np.clip(predictions, 0, None)

            mae = mean_absolute_error(y_val, predictions)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))

            print(
                f"{variant_name}: "
                f"Features={len(features)}, "
                f"MAE={mae:.4f}, "
                f"RMSE={rmse:.4f}"
            )

            fold_results.append({
                "Fold": fold_number,
                "Variant": variant_name,
                "Features": len(features),
                "MAE": mae,
                "RMSE": rmse
            })

            fold_prediction = fold_validation[
                [DATE_COL, "denominazione_regione", TARGET]
            ].copy()

            fold_prediction["Fold"] = fold_number
            fold_prediction["Variant"] = variant_name
            fold_prediction["Predicted"] = predictions

            all_predictions.append(fold_prediction)

    # Save results
    results_df = pd.DataFrame(fold_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    results_df.to_csv(
        OUTPUT_DIR / "walk_forward_results.csv",
        index=False
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "walk_forward_predictions.csv",
        index=False
    )

    # Summary across folds
    summary_df = (
        results_df.groupby("Variant")
        .agg({
            "MAE": "mean",
            "RMSE": "mean"
        })
        .reset_index()
        .sort_values("MAE")
    )

    summary_df.to_csv(
        OUTPUT_DIR / "walk_forward_summary.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nResults saved in:", OUTPUT_DIR)
    print("\nWALK-FORWARD VALIDATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
