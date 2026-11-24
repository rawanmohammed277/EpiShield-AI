"""EpiShield-AI Phase 3: One-day-ahead forecasting.
Save as Src/Forecasting/forecasting.py, then run: python forecasting.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET = "target_next_day_new_cases"
DATE, TDATE = "date", "target_date"
REGION, RNAME, CURRENT = "codice_regione", "denominazione_regione", "cases_today"
SEED = 42


def root_dir():
    for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (p / "Src" / "Feature Engineering").is_dir():
            return p
    raise FileNotFoundError("Save this script inside the project's Src/Forecasting folder.")


def load(path, label):
    d = pd.read_csv(path, low_memory=False)
    required = [DATE, TDATE, REGION, RNAME, CURRENT, TARGET]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    d[DATE] = pd.to_datetime(d[DATE], errors="coerce")
    d[TDATE] = pd.to_datetime(d[TDATE], errors="coerce")
    if d[[DATE, TDATE]].isna().any().any():
        raise ValueError(f"{label}: invalid dates")
    if not (d[TDATE] == d[DATE] + pd.Timedelta(days=1)).all():
        raise ValueError(f"{label}: target_date must be date + 1 day")
    if d.duplicated([REGION, DATE]).any():
        raise ValueError(f"{label}: duplicate region/date rows")
    if d[TARGET].isna().any() or (d[TARGET] < 0).any():
        raise ValueError(f"{label}: missing or negative target")
    return d.sort_values([REGION, DATE]).reset_index(drop=True)


def score(y, p):
    y, p = np.asarray(y, float), np.maximum(np.asarray(p, float), 0)
    nz = y != 0
    den = np.abs(y) + np.abs(p)
    valid = den != 0
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAPE_percent": float(np.mean(np.abs((y[nz]-p[nz])/y[nz]))*100) if nz.any() else None,
        "sMAPE_percent": float(np.mean(2*np.abs(y[valid]-p[valid])/den[valid])*100) if valid.any() else 0.0,
    }


def history_map(*frames):
    h = pd.concat(frames, ignore_index=True).drop_duplicates([REGION, DATE], keep="last")
    return {(r[REGION], r[DATE]): float(r[CURRENT])
            for _, r in h[[REGION, DATE, CURRENT]].iterrows() if pd.notna(r[CURRENT])}


def seasonal(df, lookup):
    out = []
    for _, r in df.iterrows():
        # Same weekday one week before target_date = origin date minus 6 days.
        v = lookup.get((r[REGION], r[TDATE] - pd.Timedelta(days=7)), np.nan)
        if pd.isna(v):
            v = r[CURRENT]
        out.append(max(float(v), 0) if pd.notna(v) else 0.0)
    return np.asarray(out)


def new_model(name):
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=200, max_depth=18,
            min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
        random_state=SEED, n_jobs=-1)


def main():
    root = root_dir()
    src = root / "Src" / "Feature Engineering" / "outputs"
    out = root / "Src" / "Forecasting" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    train = load(src / "train_dataset.csv", "Train")
    val = load(src / "validation_dataset.csv", "Validation")
    test = load(src / "test_dataset.csv", "Test")
    if train[TDATE].max() >= val[TDATE].min() or val[TDATE].max() >= test[TDATE].min():
        raise ValueError("Train/Validation/Test target periods overlap.")

    excluded = {TARGET, DATE, TDATE, REGION, RNAME}
    features = [c for c in train.columns if c not in excluded and
                pd.api.types.is_numeric_dtype(train[c])]
    if CURRENT not in features:
        raise ValueError(f"{CURRENT} must be numeric.")
    med = train[features].replace([np.inf, -np.inf], np.nan).median().fillna(0)

    def make_x(d, medians=med):
        return d[features].replace([np.inf, -np.inf], np.nan).fillna(medians).astype(float)

    xt, xv, xte = make_x(train), make_x(val), make_x(test)
    yt, yv, yte = [d[TARGET].astype(float).to_numpy() for d in (train, val, test)]
    lookup = history_map(train, val, test)

    vp = {
        "Naive": np.maximum(val[CURRENT].astype(float).to_numpy(), 0),
        "SeasonalNaive": seasonal(val, lookup),
    }
    specs = [("RandomForest", "raw"), ("RandomForest", "log1p")]
    try:
        import xgboost  # noqa
        specs += [("XGBoost", "raw"), ("XGBoost", "log1p")]
    except ImportError:
        print("XGBoost unavailable; skipping it.")

    fitted = {}
    for name, transform in specs:
        label = f"{name}_{transform}"
        model = new_model(name)
        model.fit(xt, np.log1p(yt) if transform == "log1p" else yt)
        pred = model.predict(xv)
        if transform == "log1p":
            pred = np.expm1(pred)
        vp[label] = np.maximum(pred, 0)
        fitted[label] = (name, transform)

    vm = pd.DataFrame([{"Model": k, **score(yv, v)} for k, v in vp.items()]).sort_values("MAE")
    vm.to_csv(out / "validation_model_metrics.csv", index=False)
    selected = str(vm.iloc[0]["Model"])

    # After selecting on Validation, refit candidates on Train + Validation.
    tv = pd.concat([train, val], ignore_index=True)
    med_tv = tv[features].replace([np.inf, -np.inf], np.nan).median().fillna(0)
    xtv = make_x(tv, med_tv)
    ytv = tv[TARGET].astype(float).to_numpy()
    tp = {
        "Naive": np.maximum(test[CURRENT].astype(float).to_numpy(), 0),
        "SeasonalNaive": seasonal(test, lookup),
    }
    chosen_model, chosen_transform = None, None
    for label, (name, transform) in fitted.items():
        model = new_model(name)
        model.fit(xtv, np.log1p(ytv) if transform == "log1p" else ytv)
        pred = model.predict(xte)
        if transform == "log1p":
            pred = np.expm1(pred)
        tp[label] = np.maximum(pred, 0)
        if label == selected:
            chosen_model, chosen_transform = model, transform

    tm = pd.DataFrame([{"Model": k, **score(yte, v)} for k, v in tp.items()]).sort_values("MAE")
    tm.to_csv(out / "test_model_metrics.csv", index=False)

    detail = test[[DATE, TDATE, REGION, RNAME, TARGET]].copy()
    detail.rename(columns={TARGET: "Actual"}, inplace=True)
    detail["Predicted"] = tp[selected]
    detail["Absolute_Error"] = np.abs(detail["Actual"] - detail["Predicted"])
    detail.to_csv(out / "actual_vs_predicted_test.csv", index=False)

    by_region = [{"Region": name, **score(g["Actual"], g["Predicted"]), "Rows": len(g)}
                 for name, g in detail.groupby(RNAME)]
    pd.DataFrame(by_region).sort_values("MAE").to_csv(out / "test_metrics_by_region.csv", index=False)

    if chosen_model is not None and hasattr(chosen_model, "feature_importances_"):
        pd.DataFrame({"Feature": features, "Importance": chosen_model.feature_importances_}
            ).sort_values("Importance", ascending=False).to_csv(
                out / "selected_model_feature_importance.csv", index=False)

    # Optional next-day forecast per region from the latest engineered row.
    full_path = src / "feature_engineered_dataset.csv"
    future_status = "Skipped: feature_engineered_dataset.csv not found."
    if full_path.exists():
        full = pd.read_csv(full_path, low_memory=False)
        full[DATE] = pd.to_datetime(full[DATE], errors="coerce")
        full = full.dropna(subset=[DATE])
        latest = full.sort_values(DATE).groupby(REGION, as_index=False).tail(1).copy()
        missing = [c for c in features if c not in latest.columns]
        if missing:
            future_status = "Skipped; missing features: " + ", ".join(missing)
        else:
            fx = make_x(latest, med_tv)
            if selected == "Naive":
                fp = np.maximum(latest[CURRENT].astype(float).to_numpy(), 0)
            elif selected == "SeasonalNaive":
                pseudo = latest.copy()
                pseudo[TDATE] = pseudo[DATE] + pd.Timedelta(days=1)
                fp = seasonal(pseudo, history_map(full))
            else:
                fp = chosen_model.predict(fx)
                if chosen_transform == "log1p":
                    fp = np.expm1(fp)
                fp = np.maximum(fp, 0)
            forecast = latest[[REGION, RNAME, DATE]].copy()
            forecast["Forecast_Target_Date"] = forecast[DATE] + pd.Timedelta(days=1)
            forecast["Predicted_Next_Day_Cases"] = fp
            forecast["Selected_Model"] = selected
            forecast.to_csv(out / "next_day_forecast_by_region.csv", index=False)
            future_status = "Generated next-day forecasts for all available regions."

    summary = {
        "horizon": "1 day ahead", "target": TARGET, "selected_model_by_validation_MAE": selected,
        "predictor_count": len(features), "predictors": features,
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "future_forecast_status": future_status,
        "validation_metrics": vm.to_dict(orient="records"),
        "test_metrics": tm.to_dict(orient="records")
    }
    (out / "forecasting_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    report = ["EPISHIELD-AI | PHASE 3 FORECASTING REPORT", "="*48,
        "Forecast horizon: one day ahead", f"Target: {TARGET}",
        f"Selected model by Validation MAE: {selected}", f"Predictors: {len(features)}",
        f"Rows: Train={len(train)}, Validation={len(val)}, Test={len(test)}",
        "", "VALIDATION METRICS", vm.to_string(index=False),
        "", "TEST METRICS", tm.to_string(index=False), "",
        "Method notes:",
        "- Existing Phase 2 splits were used unchanged.",
        "- Dates, target, and raw region identifiers were excluded from predictors.",
        "- Missing values were imputed using training medians.",
        "- Predictions were clipped at zero; MAPE excludes zero actuals.",
        "- Seasonal Naive uses the same weekday one week before the target date.",
        f"- Future forecast: {future_status}"]
    (out / "forecasting_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("Forecasting completed. Selected model:", selected)
    print("Outputs:", out)
    print("\nValidation:\n", vm.to_string(index=False))
    print("\nTest:\n", tm.to_string(index=False))
    print(f"\n{future_status}")


if __name__ == "__main__":
    main()
