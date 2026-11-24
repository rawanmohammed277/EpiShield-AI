
"""EpiShield-AI: Spike Feature Forecasting Experiment.

Compares four feature variants:
1. Baseline
2. Enhanced
3. Spike
4. Enhanced + Spike

The original forecasting.py and previous experiment outputs remain unchanged.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


TARGET = "target_next_day_new_cases"

DATE = "date"
TDATE = "target_date"
REGION = "codice_regione"
RNAME = "denominazione_regione"

CURRENT = "cases_today"

SEED = 42


# ============================================================
# 1. PROJECT PATHS
# ============================================================

def find_root():
    current = Path(__file__).resolve().parent

    for path in [current, *current.parents]:
        if (path / "Src" / "Feature Engineering").is_dir():
            return path

    raise FileNotFoundError(
        "Could not find project root. Place this script inside Src/Forecasting."
    )


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data(path):

    df = pd.read_csv(path, low_memory=False)

    df[DATE] = pd.to_datetime(df[DATE])
    df[TDATE] = pd.to_datetime(df[TDATE])

    expected_target_date = df[DATE] + pd.Timedelta(days=1)

    if not df[TDATE].eq(expected_target_date).all():
        raise ValueError("Target date must be exactly one day after feature date.")

    if df[TARGET].isna().any() or (df[TARGET] < 0).any():
        raise ValueError("Invalid target values.")

    return df.sort_values(
        [REGION, DATE]
    ).reset_index(drop=True)


# ============================================================
# 3. FEATURE ENGINEERING VARIANTS
# ============================================================

def add_enhanced_features(df):

    result = df.copy()

    today = result[CURRENT].clip(lower=0)
    mean7 = result["rolling_mean_7"].clip(lower=0)
    std7 = result["rolling_std_7"].clip(lower=0)
    lag1 = result["cases_lag_1"].clip(lower=0)

    safe_mean = mean7 + 1

    result["cases_today_to_mean7_ratio"] = today / safe_mean

    result["cases_today_minus_mean7"] = today - mean7

    result["volatility_ratio"] = std7 / safe_mean

    result["lag1_to_mean7_ratio"] = lag1 / safe_mean

    return result


def add_spike_features(df):

    result = df.copy()

    today = result[CURRENT].clip(lower=0)
    lag1 = result["cases_lag_1"].clip(lower=0)
    mean7 = result["rolling_mean_7"].clip(lower=0)

    # Positive increase only; negative changes become zero.
    result["positive_daily_change"] = (today - lag1).clip(lower=0)

    # Today's cases compared with the recent 7-day average.
    result["spike_ratio"] = today / (mean7 + 1)

    return result


def add_all_features(df):

    result = add_enhanced_features(df)

    result = add_spike_features(result)

    return result


# ============================================================
# 4. MODELS
# ============================================================

def make_model(name):

    if name == "RandomForest":

        return RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        )

    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1
    )


# ============================================================
# 5. EVALUATION
# ============================================================

def evaluate(y, pred, spike_threshold):

    pred = np.maximum(np.asarray(pred), 0)
    y = np.asarray(y)

    under = np.maximum(y - pred, 0)

    spikes = y >= spike_threshold

    return {
        "MAE": mean_absolute_error(y, pred),

        "RMSE": np.sqrt(
            mean_squared_error(y, pred)
        ),

        "Underprediction_Rate_%": np.mean(y > pred) * 100,

        "Mean_Underprediction": np.mean(under),

        "Max_Underprediction": np.max(under),

        "Spike_MAE": (
            mean_absolute_error(y[spikes], pred[spikes])
            if spikes.any() else None
        ),

        "Spike_Rows": int(spikes.sum())
    }


# ============================================================
# 6. FEATURE PREPARATION
# ============================================================

def get_features(df):

    excluded = {
        TARGET,
        DATE,
        TDATE,
        REGION,
        RNAME
    }

    return [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def prepare_x(df, features, medians):

    return (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .astype(float)
    )


def get_medians(df, features):

    return (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .median()
        .fillna(0)
    )


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():

    root = find_root()

    source = (
        root
        / "Src"
        / "Feature Engineering"
        / "outputs"
    )

    # Separate output folder: previous experiments stay untouched.
    output = (
        root
        / "Src"
        / "Forecasting"
        / "spike_experiment_outputs"
    )

    output.mkdir(parents=True, exist_ok=True)

    train = load_data(source / "train_dataset.csv")
    val = load_data(source / "validation_dataset.csv")
    test = load_data(source / "test_dataset.csv")

    # Verify temporal separation.
    if train[TDATE].max() >= val[TDATE].min():
        raise ValueError("Train and Validation overlap.")

    if val[TDATE].max() >= test[TDATE].min():
        raise ValueError("Validation and Test overlap.")

    # --------------------------------------------------------
    # Create four variants.
    # --------------------------------------------------------

    train_enhanced = add_enhanced_features(train)
    val_enhanced = add_enhanced_features(val)

    train_spike = add_spike_features(train)
    val_spike = add_spike_features(val)

    train_all = add_all_features(train)
    val_all = add_all_features(val)

    variants = {
        "Baseline": (train, val),
        "Enhanced": (train_enhanced, val_enhanced),
        "Spike": (train_spike, val_spike),
        "Enhanced_Spike": (train_all, val_all)
    }

    # Define high-case days using Train only.
    spike_threshold = train[TARGET].quantile(0.90)

    print("\nSpike threshold from Train:", spike_threshold)

    # --------------------------------------------------------
    # Available models.
    # --------------------------------------------------------

    try:
        import xgboost
        models = ["RandomForest", "XGBoost"]

    except ImportError:
        print("XGBoost unavailable. Using RandomForest only.")
        models = ["RandomForest"]

    # --------------------------------------------------------
    # Validation experiments.
    # --------------------------------------------------------

    validation_results = []

    candidates = {}

    for variant_name, (tr, va) in variants.items():

        features = get_features(tr)

        medians = get_medians(tr, features)

        x_train = prepare_x(tr, features, medians)
        x_val = prepare_x(va, features, medians)

        y_train = tr[TARGET].to_numpy()
        y_val = va[TARGET].to_numpy()

        print(
            f"\nRunning {variant_name} | "
            f"Features: {len(features)}"
        )

        for model_name in models:

            for transform in ["raw", "log1p"]:

                model = make_model(model_name)

                train_target = (
                    np.log1p(y_train)
                    if transform == "log1p"
                    else y_train
                )

                model.fit(x_train, train_target)

                pred = model.predict(x_val)

                if transform == "log1p":
                    pred = np.expm1(pred)

                pred = np.maximum(pred, 0)

                label = (
                    f"{variant_name}_{model_name}_{transform}"
                )

                candidates[label] = {
                    "variant": variant_name,
                    "model": model_name,
                    "transform": transform,
                    "features": features
                }

                metrics = evaluate(
                    y_val,
                    pred,
                    spike_threshold
                )

                validation_results.append({
                    "Candidate": label,
                    "Variant": variant_name,
                    "Model": model_name,
                    "Transform": transform,
                    "Feature_Count": len(features),
                    **metrics
                })

    validation_table = pd.DataFrame(validation_results)

    validation_table = validation_table.sort_values("MAE")

    validation_table.to_csv(
        output / "validation_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Select the best model per variant using Validation MAE.
    # --------------------------------------------------------

    selected = {}

    for variant_name in variants:

        subset = validation_table[
            validation_table["Variant"] == variant_name
        ].sort_values("MAE")

        selected[variant_name] = subset.iloc[0]["Candidate"]

    print("\nSELECTED MODELS")

    for variant, candidate in selected.items():
        print(f"{variant}: {candidate}")

    # --------------------------------------------------------
    # Final Test evaluation.
    # Train final models on Train + Validation.
    # --------------------------------------------------------

    combined = pd.concat(
        [train, val],
        ignore_index=True
    )

    combined_variants = {
        "Baseline": combined,
        "Enhanced": add_enhanced_features(combined),
        "Spike": add_spike_features(combined),
        "Enhanced_Spike": add_all_features(combined)
    }

    test_variants = {
        "Baseline": test,
        "Enhanced": add_enhanced_features(test),
        "Spike": add_spike_features(test),
        "Enhanced_Spike": add_all_features(test)
    }

    y_combined = combined[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    final_results = []

    for variant_name in variants:

        candidate = selected[variant_name]

        info = candidates[candidate]

        model_name = info["model"]
        transform = info["transform"]
        features = info["features"]

        train_data = combined_variants[variant_name]
        test_data = test_variants[variant_name]

        medians = get_medians(train_data, features)

        x_train = prepare_x(train_data, features, medians)
        x_test = prepare_x(test_data, features, medians)

        model = make_model(model_name)

        target = (
            np.log1p(y_combined)
            if transform == "log1p"
            else y_combined
        )

        model.fit(x_train, target)

        pred = model.predict(x_test)

        if transform == "log1p":
            pred = np.expm1(pred)

        pred = np.maximum(pred, 0)

        metrics = evaluate(
            y_test,
            pred,
            spike_threshold
        )

        final_results.append({
            "Variant": variant_name,
            "Selected_Model": candidate,
            "Feature_Count": len(features),
            **metrics
        })

        # Save detailed predictions.
        details = test[
            [DATE, TDATE, REGION, RNAME, TARGET]
        ].copy()

        details.rename(
            columns={TARGET: "Actual"},
            inplace=True
        )

        details["Predicted"] = pred

        details["Absolute_Error"] = abs(
            details["Actual"] - details["Predicted"]
        )

        details["Underprediction"] = np.maximum(
            details["Actual"] - details["Predicted"],
            0
        )

        details.to_csv(
            output / f"{variant_name.lower()}_test_predictions.csv",
            index=False
        )

    # --------------------------------------------------------
    # Save final results.
    # --------------------------------------------------------

    final_table = pd.DataFrame(final_results)

    final_table.to_csv(
        output / "test_comparison.csv",
        index=False
    )

    summary = {
        "spike_threshold_from_train_only": float(spike_threshold),

        "selected_models": selected,

        "validation_results": validation_table.to_dict("records"),

        "test_results": final_table.to_dict("records")
    }

    (output / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(validation_table.to_string(index=False))

    print("\n" + "=" * 70)
    print("FINAL TEST COMPARISON")
    print("=" * 70)

    print(final_table.to_string(index=False))

    print("\nResults saved to:")
    print(output)


if __name__ == "__main__":
    main()
