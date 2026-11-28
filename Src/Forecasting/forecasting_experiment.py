
"""EpiShield-AI: Experimental Forecasting Comparison.
Original forecasting.py and outputs remain unchanged.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET = "target_next_day_new_cases"
DATE, TDATE = "date", "target_date"
REGION, RNAME = "codice_regione", "denominazione_regione"
CURRENT = "cases_today"
SEED = 42


def find_root():
    for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (p / "Src" / "Feature Engineering").is_dir():
            return p
    raise FileNotFoundError("Place this script inside Src/Forecasting.")


def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df[DATE] = pd.to_datetime(df[DATE])
    df[TDATE] = pd.to_datetime(df[TDATE])

    if not (df[TDATE] == df[DATE] + pd.Timedelta(days=1)).all():
        raise ValueError("Target date must be one day after feature date.")
    if df[TARGET].isna().any() or (df[TARGET] < 0).any():
        raise ValueError("Invalid target values.")

    return df.sort_values([REGION, DATE]).reset_index(drop=True)


def add_features(df):
    result = df.copy()

    today = result[CURRENT].clip(lower=0)
    mean7 = result["rolling_mean_7"].clip(lower=0)
    std7 = result["rolling_std_7"].clip(lower=0)
    lag1 = result["cases_lag_1"].clip(lower=0)

    # Additional features based only on information available today.
    safe_mean = mean7 + 1

    result["cases_today_to_mean7_ratio"] = today / safe_mean
    result["cases_today_minus_mean7"] = today - mean7
    result["volatility_ratio"] = std7 / safe_mean
    result["lag1_to_mean7_ratio"] = lag1 / safe_mean

    return result


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


def evaluate(y, pred, spike_threshold):
    pred = np.maximum(np.asarray(pred), 0)
    y = np.asarray(y)

    under = np.maximum(y - pred, 0)
    spikes = y >= spike_threshold

    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "Underprediction_Rate_%": np.mean(y > pred) * 100,
        "Mean_Underprediction": np.mean(under),
        "Max_Underprediction": np.max(under),
        "Spike_MAE": (
            mean_absolute_error(y[spikes], pred[spikes])
            if spikes.any() else None
        ),
        "Spike_Rows": int(spikes.sum())
    }


def get_features(df):
    excluded = {TARGET, DATE, TDATE, REGION, RNAME}

    return [
        c for c in df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def prepare_x(df, features, medians):
    return (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .astype(float)
    )


def main():
    root = find_root()

    source = root / "Src" / "Feature Engineering" / "outputs"
    output = root / "Src" / "Forecasting" / "experiment_outputs"
    output.mkdir(parents=True, exist_ok=True)

    train = load_data(source / "train_dataset.csv")
    val = load_data(source / "validation_dataset.csv")
    test = load_data(source / "test_dataset.csv")

    if train[TDATE].max() >= val[TDATE].min():
        raise ValueError("Train and Validation overlap.")
    if val[TDATE].max() >= test[TDATE].min():
        raise ValueError("Validation and Test overlap.")

    train_e = add_features(train)
    val_e = add_features(val)
    test_e = add_features(test)

    baseline_features = get_features(train)
    enhanced_features = get_features(train_e)

    # Define high-case days using Train data only.
    spike_threshold = train[TARGET].quantile(0.90)

    try:
        import xgboost
        models = ["RandomForest", "XGBoost"]
    except ImportError:
        print("XGBoost unavailable. Using RandomForest only.")
        models = ["RandomForest"]

    validation_results = []
    candidates = {}

    variants = [
        ("Baseline", train, val, baseline_features),
        ("Enhanced", train_e, val_e, enhanced_features)
    ]

    for variant, tr, va, features in variants:
        medians = (
            tr[features]
            .replace([np.inf, -np.inf], np.nan)
            .median()
            .fillna(0)
        )

        x_train = prepare_x(tr, features, medians)
        x_val = prepare_x(va, features, medians)

        y_train = tr[TARGET].to_numpy()
        y_val = va[TARGET].to_numpy()

        for model_name in models:
            for transform in ["raw", "log1p"]:
                model = make_model(model_name)

                train_target = (
                    np.log1p(y_train)
                    if transform == "log1p" else y_train
                )

                model.fit(x_train, train_target)

                pred = model.predict(x_val)

                if transform == "log1p":
                    pred = np.expm1(pred)

                pred = np.maximum(pred, 0)

                label = f"{variant}_{model_name}_{transform}"

                candidates[label] = (
                    variant, model_name, transform, features
                )

                validation_results.append({
                    "Candidate": label,
                    "Variant": variant,
                    "Model": model_name,
                    "Transform": transform,
                    **evaluate(y_val, pred, spike_threshold)
                })

    validation_table = pd.DataFrame(validation_results)
    validation_table = validation_table.sort_values("MAE")

    validation_table.to_csv(
        output / "validation_experiment_metrics.csv",
        index=False
    )

    # Select the best candidate per variant using Validation MAE.
    selected = {}

    for variant in ["Baseline", "Enhanced"]:
        subset = validation_table[
            validation_table["Variant"] == variant
        ].sort_values("MAE")

        selected[variant] = subset.iloc[0]["Candidate"]

    final_results = []

    combined = pd.concat([train, val], ignore_index=True)
    combined_e = add_features(combined)

    y_combined = combined[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    for variant, train_data, test_data in [
        ("Baseline", combined, test),
        ("Enhanced", combined_e, test_e)
    ]:
        candidate = selected[variant]
        _, model_name, transform, features = candidates[candidate]

        medians = (
            train_data[features]
            .replace([np.inf, -np.inf], np.nan)
            .median()
            .fillna(0)
        )

        x_train = prepare_x(train_data, features, medians)
        x_test = prepare_x(test_data, features, medians)

        model = make_model(model_name)

        target = (
            np.log1p(y_combined)
            if transform == "log1p" else y_combined
        )

        model.fit(x_train, target)

        pred = model.predict(x_test)

        if transform == "log1p":
            pred = np.expm1(pred)

        pred = np.maximum(pred, 0)

        final_results.append({
            "Variant": variant,
            "Selected_Model": candidate,
            **evaluate(y_test, pred, spike_threshold)
        })

        details = test[
            [DATE, TDATE, REGION, RNAME, TARGET]
        ].copy()

        details.rename(columns={TARGET: "Actual"}, inplace=True)
        details["Predicted"] = pred
        details["Underprediction"] = np.maximum(
            details["Actual"] - pred, 0
        )

        details.to_csv(
            output / f"{variant.lower()}_test_predictions.csv",
            index=False
        )

    final_table = pd.DataFrame(final_results)

    final_table.to_csv(
        output / "test_experiment_comparison.csv",
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

    print("\nVALIDATION RESULTS")
    print(validation_table.to_string(index=False))

    print("\nSELECTED MODELS")
    print(selected)

    print("\nFINAL TEST COMPARISON")
    print(final_table.to_string(index=False))

    print("\nResults saved to:", output)


if __name__ == "__main__":
    main()
