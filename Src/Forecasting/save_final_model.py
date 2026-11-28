
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

import forecasting_experiment as fe


def main():

    print("PREPARING FINAL MODEL")

    root = fe.find_root()

    source_dir = root / "Src" / "Feature Engineering" / "outputs"

    output_dir = root / "Src" / "Forecasting" / "final_model_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Train and Validation only
    train = fe.load_data(source_dir / "train_dataset.csv")
    validation = fe.load_data(source_dir / "validation_dataset.csv")

    if train[fe.TDATE].max() >= validation[fe.TDATE].min():
        raise ValueError("Train and Validation overlap.")

    # Combine Train + Validation
    combined = pd.concat(
        [train, validation],
        ignore_index=True
    )

    # Create Enhanced features
    combined = fe.add_features(combined)

    # Get the same Enhanced feature list
    features = fe.get_features(combined)

    if len(features) != 45:
        raise ValueError(
            f"Expected 45 features, but found {len(features)}"
        )

    # Calculate missing-value replacements using Train + Validation
    medians = (
        combined[features]
        .replace([np.inf, -np.inf], np.nan)
        .median()
        .fillna(0)
    )

    X_train = fe.prepare_x(combined, features, medians)

    y_train = combined[fe.TARGET].to_numpy()

    if (y_train < 0).any():
        raise ValueError("Negative target values found.")

    # Create the selected XGBoost model
    model = fe.make_model("XGBoost")

    # Train with log1p target transformation
    model.fit(X_train, np.log1p(y_train))

    # Save model and preprocessing information together
    artifact = {
        "model": model,
        "features": features,
        "medians": medians.to_dict(),
        "target_transform": "log1p",
        "prediction_inverse_transform": "expm1",
        "prediction_clip_min": 0,
        "model_name": "Enhanced_XGBoost_log1p",
        "training_rows": len(combined)
    }

    model_path = output_dir / "epishield_enhanced_xgboost.joblib"

    joblib.dump(artifact, model_path)

    # Save readable metadata
    metadata = {
        "model_name": "Enhanced_XGBoost_log1p",
        "variant": "Enhanced",
        "algorithm": "XGBoost",
        "target_transform": "log1p",
        "features_count": len(features),
        "features": features,
        "training_rows": len(combined),
        "training_period": {
            "start": str(combined[fe.DATE].min().date()),
            "end": str(combined[fe.DATE].max().date())
        },
        "parameters": {
            "n_estimators": 350,
            "max_depth": 6,
            "learning_rate": 0.04,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "objective": "reg:squarederror",
            "random_state": 42
        },
        "artifact_file": model_path.name
    }

    metadata_path = output_dir / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\nFINAL MODEL SAVED SUCCESSFULLY")
    print("--------------------------------")
    print("Model:", model_path)
    print("Metadata:", metadata_path)
    print("Features:", len(features))
    print("Training rows:", len(combined))
    print("Target transformation: log1p")
    print("\nFORECASTING MODEL ARTIFACT READY")


if __name__ == "__main__":
    main()
