
import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

ROOT = Path("Src")

test_path = ROOT / "Data Engineering" / "Data_Cleaning" / "output" / "test_dataset.csv"
population_path = ROOT / "Data Engineering" / "outputs" / "eda" / "population_data.csv"
metadata_path = ROOT / "Forecasting" / "final_model_artifacts" / "model_metadata.json"


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def number(value):
    try:
        if value is None or value.strip() == "":
            return None
        return float(value)
    except (ValueError, AttributeError):
        return None


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None

    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return values[lower] * (1 - fraction) + values[upper] * fraction


df, columns = read_csv(test_path)
pop, pop_columns = read_csv(population_path)

with open(metadata_path, "r", encoding="utf-8-sig") as f:
    metadata = json.load(f)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", len(df))
print("Columns:", len(columns))

dates = [r["date"] for r in df if r.get("date")]
regions = set(r["codice_regione"] for r in df)

print("Date range:", min(dates), "to", max(dates))
print("Regions:", len(regions))

print("\n========== DUPLICATES ==========")

keys = [
    (r["date"], r["codice_regione"])
    for r in df
]

print("Duplicate region-date rows:", len(keys) - len(set(keys)))

indicators = [
    "cases_today",
    "cases_per_100k",
    "rolling_mean_7_per_100k",
    "growth_rate_7",
    "rolling_std_7",
    "hospitalized",
    "icu",
    "hospitalized_per_100k",
    "icu_per_100k"
]

print("\n========== MISSING VALUES ==========")

for col in indicators:
    if col in columns:
        missing = sum(
            1 for r in df
            if number(r.get(col)) is None
        )
        print(col, "Missing:", missing)

print("\n========== INDICATOR DISTRIBUTIONS ==========")

for col in indicators:
    if col not in columns:
        continue

    values = [
        number(r.get(col))
        for r in df
    ]
    values = [v for v in values if v is not None]

    if not values:
        continue

    print("\n", col)
    print("Count:", len(values))
    print("Mean:", round(mean(values), 4))
    print("Min:", min(values))
    print("25%:", round(percentile(values, 0.25), 4))
    print("Median:", round(percentile(values, 0.50), 4))
    print("75%:", round(percentile(values, 0.75), 4))
    print("90%:", round(percentile(values, 0.90), 4))
    print("95%:", round(percentile(values, 0.95), 4))
    print("Max:", max(values))

print("\n========== HEALTHCARE VALIDATION ==========")

hospitalized = [number(r["hospitalized"]) for r in df]
icu = [number(r["icu"]) for r in df]

print("Negative hospitalized:",
      sum(v < 0 for v in hospitalized if v is not None))

print("Negative ICU:",
      sum(v < 0 for v in icu if v is not None))

print("ICU greater than hospitalized:",
      sum(
          1 for h, i in zip(hospitalized, icu)
          if h is not None and i is not None and i > h
      ))

print("\n========== POPULATION VALIDATION ==========")

pop_agg = defaultdict(float)

for row in pop:
    code = row["region_code_norm"]
    population = number(row["population_total"])

    if population is not None:
        pop_agg[code] += population

print("Population regions:", len(pop_agg))

for code, total in sorted(pop_agg.items()):
    print(code, round(total))

print("\n========== MODEL FEATURE VALIDATION ==========")

model_features = set(metadata["features"])
dataset_features = set(columns)

print("Missing from dataset:")
print(sorted(model_features - dataset_features))

print("\nExtra in dataset:")
print(sorted(dataset_features - model_features))

print("\n========== END OF AUDIT ==========")
