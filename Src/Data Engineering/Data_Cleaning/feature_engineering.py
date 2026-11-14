# feature_engineering.py
# Step 2 من EpiShield: Feature Engineering
# بيقرا الداتا النضيفة من فولدر clean، وبيطلع في فولدر output:
#   feature_engineered_dataset.csv  (كل الداتا بالـ features والـ target)
#   train_dataset.csv / validation_dataset.csv / test_dataset.csv
#   feature_description.csv  (شرح كل feature)
#   feature_validation_report.txt  (نتيجة الفحوصات)

import numpy as np
import pandas as pd
from pathlib import Path

CLEAN = Path("clean")    # فولدر الداتا النضيفة (غيّريه لو مكانه مختلف)
OUT = Path("output")     # هنا هتطلع النتايج
OUT.mkdir(exist_ok=True)

HORIZON = 7              # بنتوقع بعد 7 أيام
report = []              # هنكتب فيه نتايج الفحوصات


def log(text):
    print(text)
    report.append(text)


# ============ 1) Data Preparation: تجهيز الداتا ============
df = pd.read_csv(CLEAN / "regioni_clean.csv",
                 dtype={"codice_regione": str}, parse_dates=["date"])
lookup = pd.read_csv(CLEAN / "regioni_lookup.csv", dtype={"codice_regione": str})

# بنختار الأعمدة اللي محتاجينها بس
df = df[["date", "codice_regione", "denominazione_regione",
         "nuovi_positivi", "totale_ospedalizzati", "terapia_intensiva"]].copy()
df = df.rename(columns={"totale_ospedalizzati": "hospitalized",
                        "terapia_intensiva": "icu"})

# نضيف عدد سكان كل إقليم (من ملف الأقاليم)
df = df.merge(lookup[["codice_regione", "population_total"]],
              on="codice_regione", how="left")

# نرتب: كل إقليم بالتاريخ
df = df.sort_values(["codice_regione", "date"]).reset_index(drop=True)

# الحالات الجديدة السالبة (14 صف، تصحيحات من المصدر) بنخليها صفر في نسخة النمذجة
# الأصل في الملف النضيف مش بيتغير
df["new_cases"] = df["nuovi_positivi"].clip(lower=0)

log("=== Data Preparation ===")
log(f"عدد الصفوف: {len(df)}")
log(f"عدد الأقاليم: {df['codice_regione'].nunique()}")
log(f"الفترة: {df['date'].min().date()} لـ {df['date'].max().date()}")
log(f"تكرار نفس الإقليم في نفس اليوم: {df.duplicated(['codice_regione', 'date']).sum()}")
gaps = (df.groupby("codice_regione")["date"].diff().dropna() != pd.Timedelta(days=1)).sum()
log(f"أيام ناقصة (فجوات) جوا أي إقليم: {gaps}")
log(f"قيم سالبة اتحولت لصفر: {(df['nuovi_positivi'] < 0).sum()}")
log(f"إقليم من غير سكان: {df['population_total'].isna().sum()}")


# ============ 2) الـ Features ============
# كل صف = "يوم التوقع" t. كل الـ features بتتحسب من بيانات لحد يوم t (بما فيه النهارده)
# والـ target هو حالات يوم t+7. يعني مفيش أي معلومة من المستقبل في الـ features.
def make_features(d):
    d = d.sort_values(["codice_regione", "date"]).reset_index(drop=True)
    g = d.groupby("codice_regione")["new_cases"]   # كل حاجة بتتحسب لكل إقليم لوحده

    # --- معلومات الوقت ---
    d["day"] = d["date"].dt.day
    d["day_of_week"] = d["date"].dt.day_of_week      # 0 = الإثنين
    d["month"] = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter
    d["year"] = d["date"].dt.year
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)

    # --- Lag: حالات الأيام اللي فاتت ---
    d["cases_today"] = d["new_cases"]
    for k in [1, 2, 3, 7, 14]:
        d[f"cases_lag_{k}"] = g.shift(k)

    # --- Rolling: متوسطات آخر كام يوم (بتشمل النهارده) ---
    for w in [7, 14, 30]:
        d[f"rolling_mean_{w}"] = g.transform(lambda s: s.rolling(w).mean())
    d["rolling_std_7"] = g.transform(lambda s: s.rolling(7).std())
    d["rolling_max_7"] = g.transform(lambda s: s.rolling(7).max())

    # --- مؤشرات إضافية ---
    d["daily_change"] = d["cases_today"] - d["cases_lag_1"]
    # معدل النمو: متوسط آخر 7 أيام مقارنة بمتوسط الـ 7 أيام اللي قبلهم
    # بنزود 1 في المقام عشان ما نقسمش على صفر
    prev_mean_7 = g.transform(lambda s: s.shift(7).rolling(7).mean())
    d["growth_rate_7"] = (d["rolling_mean_7"] - prev_mean_7) / (prev_mean_7 + 1)

    # الحالات لكل 100 ألف نسمة
    d["cases_per_100k"] = d["cases_today"] / d["population_total"] * 100000
    d["rolling_mean_7_per_100k"] = d["rolling_mean_7"] / d["population_total"] * 100000
    # ضغط المستشفيات لكل 100 ألف نسمة
    d["hospitalized_per_100k"] = d["hospitalized"] / d["population_total"] * 100000
    d["icu_per_100k"] = d["icu"] / d["population_total"] * 100000
    return d


FEATURES = ["day", "day_of_week", "month", "quarter", "year", "is_weekend",
            "cases_today", "cases_lag_1", "cases_lag_2", "cases_lag_3",
            "cases_lag_7", "cases_lag_14",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
            "rolling_std_7", "rolling_max_7",
            "daily_change", "growth_rate_7",
            "cases_per_100k", "rolling_mean_7_per_100k",
            "hospitalized", "icu", "hospitalized_per_100k", "icu_per_100k"]

full = make_features(df)

# --- الـ Target: حالات بعد 7 أيام، لكل إقليم لوحده ---
full["target_cases_7d"] = full.groupby("codice_regione")["new_cases"].shift(-HORIZON)


# ============ 3) فحص الـ Features ============
log("\n=== Feature Validation ===")

# أ) الصفوف الناقصة الطبيعية (أول 29 يوم في كل إقليم عشان الـ rolling 30، وآخر 7 أيام عشان الـ target)
nan_before = full[FEATURES + ["target_cases_7d"]].isna().any(axis=1).sum()
log(f"صفوف فيها قيم ناقصة (طبيعي بسبب Lag/Rolling/Target): {nan_before}")
log(f"المتوقع: {df['codice_regione'].nunique() * (29 + HORIZON)}")

# ب) فحص مستقل: نحسب نفس القيم بطريقة تانية (بالتاريخ) ونقارن، عشان نتأكد إن مفيش اختلاط بين الأقاليم
s = full.set_index(["codice_regione", "date"])["new_cases"]
sample = full.dropna(subset=FEATURES + ["target_cases_7d"]).sample(3000, random_state=1)


def value_at(days_shift):
    idx = pd.MultiIndex.from_arrays([sample["codice_regione"],
                                     sample["date"] + pd.Timedelta(days=days_shift)])
    return s.reindex(idx).values


log(f"cases_lag_7 مطابق للقيمة الحقيقية قبل 7 أيام: {np.allclose(sample['cases_lag_7'], value_at(-7))}")
log(f"cases_lag_14 مطابق: {np.allclose(sample['cases_lag_14'], value_at(-14))}")
log(f"target مطابق لحالات يوم t+7: {np.allclose(sample['target_cases_7d'], value_at(HORIZON))}")
manual_mean_7 = np.mean([value_at(-k) for k in range(7)], axis=0)
log(f"rolling_mean_7 مطابق للحساب اليدوي: {np.allclose(sample['rolling_mean_7'], manual_mean_7)}")

# ج) اختبار عدم التسريب: لو قطعنا الداتا عند تاريخ معين وحسبنا الـ features
# المفروض نفس الـ features في الأيام اللي قبل القطع تطلع بنفس القيم بالظبط
# (يعني الـ features مش بتستخدم أي حاجة من بعد اليوم)
cutoff = pd.Timestamp("2022-06-30")
cut = make_features(df[df["date"] <= cutoff].copy())
cmp_cols = ["codice_regione", "date"] + FEATURES
merged = cut[cmp_cols].merge(full[cmp_cols], on=["codice_regione", "date"],
                             suffixes=("_cut", "_full"))
same = True
for c in FEATURES:
    a, b = merged[c + "_cut"].to_numpy(float), merged[c + "_full"].to_numpy(float)
    if not np.allclose(a, b, equal_nan=True):
        same = False
        log(f"  تحذير: العمود {c} اتغير لما قطعنا الداتا")
log(f"اختبار القطع (مفيش تسريب من المستقبل): {'نجح' if same else 'فشل'}")

# د) قيم غير منطقية
log(f"أقل قيمة حالات اليوم: {full['cases_today'].min()}")
num = full[FEATURES + ["target_cases_7d"]].select_dtypes("number")
log(f"قيم لانهائية (inf): {int(np.isinf(num.to_numpy(float)).sum())}")


# ============ 4) الداتا النهائية والتقسيم الزمني ============
final = full.dropna(subset=FEATURES + ["target_cases_7d"]).copy()

# Encoding للإقليم: عمود 0/1 لكل إقليم (مش رقم، عشان الموديل ما يفتكرش إن إقليم 10 أكبر من 2)
dummies = pd.get_dummies(final["codice_regione"], prefix="region", dtype=int)
final = pd.concat([final, dummies], axis=1)

ID_COLS = ["date", "codice_regione", "denominazione_regione"]
X_COLS = FEATURES + list(dummies.columns)
final = final[ID_COLS + X_COLS + ["target_cases_7d"]]

# التقسيم بالتاريخ: 70% قديم للتدريب، 15% للتحقق، 15% أحدث للاختبار
dates = np.sort(final["date"].unique())
t1 = pd.Timestamp(dates[int(len(dates) * 0.70)])
t2 = pd.Timestamp(dates[int(len(dates) * 0.85)])
gap = pd.Timedelta(days=HORIZON)

# فجوة 7 أيام في آخر التدريب والتحقق، عشان الـ target بتاع آخر أيام التدريب
# (بعد 7 أيام) ما يقعش جوا فترة التحقق
train = final[final["date"] <= t1 - gap]
valid = final[(final["date"] > t1) & (final["date"] <= t2 - gap)]
test = final[final["date"] > t2]

log("\n=== التقسيم ===")
for name, part in [("train", train), ("validation", valid), ("test", test)]:
    log(f"{name}: {len(part)} صف | من {part['date'].min().date()} لـ {part['date'].max().date()}"
        f" | أقاليم: {part['codice_regione'].nunique()}")
log(f"مفيش تداخل في التواريخ: {train['date'].max() < valid['date'].min() and valid['date'].max() < test['date'].min()}")
log(f"آخر target في التدريب بيخلص قبل بداية التحقق: "
    f"{(train['date'].max() + gap) <= t1}")
log(f"\nالداتا النهائية: {final.shape} | features: {len(X_COLS)} | قيم ناقصة: {int(final.isna().sum().sum())}")


# ============ 5) حفظ الملفات ============
final.to_csv(OUT / "feature_engineered_dataset.csv", index=False)
train.to_csv(OUT / "train_dataset.csv", index=False)
valid.to_csv(OUT / "validation_dataset.csv", index=False)
test.to_csv(OUT / "test_dataset.csv", index=False)

desc = [
    ("target_cases_7d", "Target", "الحالات الجديدة بعد 7 أيام (حالات يوم t+7)", "shift(-7) لكل إقليم"),
    ("day", "Time", "رقم اليوم في الشهر", "من التاريخ"),
    ("day_of_week", "Time", "يوم الأسبوع (0 الإثنين لـ 6 الأحد)", "من التاريخ"),
    ("month", "Time", "الشهر", "من التاريخ"),
    ("quarter", "Time", "الربع السنوي", "من التاريخ"),
    ("year", "Time", "السنة", "من التاريخ"),
    ("is_weekend", "Time", "هل اليوم سبت أو أحد", "day_of_week >= 5"),
    ("cases_today", "Lag", "حالات اليوم (يوم التوقع)", "new_cases"),
    ("cases_lag_1", "Lag", "حالات إمبارح", "shift(1) لكل إقليم"),
    ("cases_lag_2", "Lag", "حالات من يومين", "shift(2)"),
    ("cases_lag_3", "Lag", "حالات من 3 أيام", "shift(3)"),
    ("cases_lag_7", "Lag", "حالات من أسبوع", "shift(7)"),
    ("cases_lag_14", "Lag", "حالات من أسبوعين", "shift(14)"),
    ("rolling_mean_7", "Rolling", "متوسط آخر 7 أيام (بيشمل النهارده)", "rolling(7).mean()"),
    ("rolling_mean_14", "Rolling", "متوسط آخر 14 يوم", "rolling(14).mean()"),
    ("rolling_mean_30", "Rolling", "متوسط آخر 30 يوم", "rolling(30).mean()"),
    ("rolling_std_7", "Rolling", "تذبذب الحالات في آخر 7 أيام", "rolling(7).std()"),
    ("rolling_max_7", "Rolling", "أعلى حالات في آخر 7 أيام", "rolling(7).max()"),
    ("daily_change", "Additional", "الفرق بين حالات اليوم وإمبارح", "cases_today - cases_lag_1"),
    ("growth_rate_7", "Additional", "معدل نمو متوسط 7 أيام مقارنة بالـ 7 أيام اللي قبلهم",
     "(mean7 - prev_mean7) / (prev_mean7 + 1)"),
    ("cases_per_100k", "Additional", "حالات اليوم لكل 100 ألف نسمة", "cases_today / population * 100000"),
    ("rolling_mean_7_per_100k", "Additional", "متوسط 7 أيام لكل 100 ألف نسمة", "rolling_mean_7 / population * 100000"),
    ("hospitalized", "Additional", "إجمالي المحجوزين في المستشفى اليوم", "totale_ospedalizzati"),
    ("icu", "Additional", "الموجودين في العناية المركزة اليوم", "terapia_intensiva"),
    ("hospitalized_per_100k", "Additional", "المحجوزين لكل 100 ألف نسمة", "hospitalized / population * 100000"),
    ("icu_per_100k", "Additional", "العناية المركزة لكل 100 ألف نسمة", "icu / population * 100000"),
    ("region_XX", "Region", "عمود 0/1 لكل إقليم (21 عمود)، XX كود الإقليم", "get_dummies"),
]
pd.DataFrame(desc, columns=["feature", "type", "meaning", "how_computed"]).to_csv(
    OUT / "feature_description.csv", index=False, encoding="utf-8-sig")

(OUT / "feature_validation_report.txt").write_text("\n".join(report), encoding="utf-8")
print("\nخلصنا. النتايج في فولدر output")
