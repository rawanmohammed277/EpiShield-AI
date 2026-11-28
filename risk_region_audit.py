
import csv
from pathlib import Path
from collections import defaultdict

path = Path("Src/Data Engineering/Data_Cleaning/output/test_dataset.csv")

with open(path, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

regions = defaultdict(list)

for row in rows:
    regions[row["denominazione_regione"]].append(row)

print("\n========== REGIONAL ANALYSIS ==========")

for region, data in sorted(regions.items()):
    avg_cases = sum(float(r["cases_per_100k"]) for r in data) / len(data)
    avg_growth = sum(float(r["growth_rate_7"]) for r in data) / len(data)
    avg_hospital = sum(float(r["hospitalized_per_100k"]) for r in data) / len(data)
    avg_icu = sum(float(r["icu_per_100k"]) for r in data) / len(data)

    print("\nRegion:", region)
    print("Average Cases:", round(avg_cases, 3))
    print("Average Growth:", round(avg_growth, 3))
    print("Average Hospitalized:", round(avg_hospital, 3))
    print("Average ICU:", round(avg_icu, 3))

print("\n========== TOP 15 GROWTH VALUES ==========")

extreme = sorted(
    rows,
    key=lambda r: float(r["growth_rate_7"]),
    reverse=True
)[:15]

for r in extreme:
    print(
        r["date"],
        "|", r["denominazione_regione"],
        "| Cases:", r["cases_today"],
        "| Growth:", r["growth_rate_7"]
    )
