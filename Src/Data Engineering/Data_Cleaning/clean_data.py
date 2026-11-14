# clean_data.py
# تنضيف داتا EpiShield
# بيقرا الملفات الخام من فولدر raw وبيطلع الملفات النضيفة في فولدر clean
# مهم: الأرقام الأصلية مش بتتغير، بنضيف بس أعمدة تحذير (flags) للقيم الغريبة

import pandas as pd
from pathlib import Path

RAW = Path("raw")      # فولدر الملفات الخام (regioni, nazionale, popolazione, note)
CLEAN = Path("clean")  # هنا هتطلع الملفات النضيفة
CLEAN.mkdir(exist_ok=True)


def clean_text(s):
    # يشيل التابات والمسافات الزيادة، والفاضي يبقى NaN
    s = s.astype("string")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.replace("", pd.NA)


def to_int(df, cols):
    # الأرقام الصحيحة نخليها Int64 عشان القيم الناقصة تفضل ناقصة (مش 0)
    for c in cols:
        df[c] = df[c].astype("Int64")
    return df


# أسماء الأقاليم: ملف الحالات فيه P.A. Bolzano وملف السكان فيه Bolzano
# هنوحدها على اسم ملف السكان
NAME_FIX = {"P.A. Bolzano": "Bolzano", "P.A. Trento": "Trento"}


# ============ 1) ملف الأقاليم (الحالات اليومية لكل إقليم) ============
r = pd.read_csv(RAW / "regioni.csv", dtype={"codice_regione": "string"})
print("الأقاليم قبل:", r.shape)

# التاريخ: بناخد اليوم بس (الساعة كانت 17:00 في ناس و18:00 في ناس)
r["date"] = pd.to_datetime(r["data"]).dt.strftime("%Y-%m-%d")
r = r.drop(columns=["data", "stato"])  # stato كله ITA، مالوش لازمة

# كود الإقليم نص من رقمين (عشان الصفر اللي في الأول ما يضيعش)
r["codice_regione"] = r["codice_regione"].str.zfill(2)

# توحيد اسم الإقليم
r["denominazione_regione"] = clean_text(r["denominazione_regione"]).replace(NAME_FIX)

# الإحداثيات: فيه رقم واحد مكتوب بكسور زيادة، نقربهم كلهم لنفس الشكل
r["lat"] = r["lat"].round(8)
r["long"] = r["long"].round(8)

# NUTS ناقصة في أول فترة (لحد يناير 2021)، بس هي ثابتة لكل إقليم
# فنملاها من باقي صفوف نفس الإقليم
for c in ["codice_nuts_1", "codice_nuts_2"]:
    r[c] = r.groupby("codice_regione")[c].transform(lambda s: s.ffill().bfill())

# الأعمدة النصية (الملاحظات)
for c in ["note", "note_test", "note_casi"]:
    r[c] = clean_text(r[c])

# باقي الأعمدة أرقام صحيحة
text_cols = ["codice_regione", "denominazione_regione", "date", "codice_nuts_1",
             "codice_nuts_2", "note", "note_test", "note_casi", "lat", "long"]
count_cols = [c for c in r.columns if c not in text_cols]
r = to_int(r, count_cols)

# ترتيب: كل إقليم بالتاريخ
r = r.sort_values(["codice_regione", "date"]).reset_index(drop=True)

# ---- أعمدة التحذير (من غير ما نغير الأرقام) ----
# 1) حالات جديدة سالبة (تصحيح من المصدر)
r["flag_negative_new_cases"] = (r["nuovi_positivi"] < 0).fillna(False).astype(bool)

# 2) رقم تراكمي نزل عن اليوم اللي قبله (المفروض التراكمي ما ينزلش)
dec = pd.Series(False, index=r.index)
for c in ["totale_casi", "deceduti", "tamponi", "dimessi_guariti"]:
    d = r[c].astype(float).groupby(r["codice_regione"]).diff()
    dec = dec | (d < 0)
r["flag_cumulative_decrease"] = dec.fillna(False).astype(bool)

# 3) المعادلات اللي لازم تتطابق بس مش متطابقة
mismatch1 = (r["totale_ospedalizzati"] + r["isolamento_domiciliare"]) != r["totale_positivi"]
mismatch2 = (r["totale_positivi"] + r["dimessi_guariti"] + r["deceduti"]) != r["totale_casi"]
r["flag_identity_mismatch"] = (mismatch1 | mismatch2).fillna(False).astype(bool)

# التاريخ أول عمود
r = r[["date"] + [c for c in r.columns if c != "date"]]
r.to_csv(CLEAN / "regioni_clean.csv", index=False, encoding="utf-8-sig")
print("الأقاليم بعد:", r.shape)


# ============ 2) ملف الوطني (إيطاليا كلها) ============
n = pd.read_csv(RAW / "nazionale.csv")
print("الوطني قبل:", n.shape)

n["date"] = pd.to_datetime(n["data"]).dt.strftime("%Y-%m-%d")
n = n.drop(columns=["data", "stato"])

# نشيل الأعمدة الفاضية تمامًا
empty_cols = [c for c in n.columns if n[c].isna().all()]
print("أعمدة فاضية اتشالت من الوطني:", empty_cols)
n = n.drop(columns=empty_cols)

n["note"] = clean_text(n["note"])
count_cols = [c for c in n.columns if c not in ["date", "note"]]
n = to_int(n, count_cols)

n = n.sort_values("date").reset_index(drop=True)
n = n[["date"] + [c for c in n.columns if c != "date"]]
n.to_csv(CLEAN / "nazionale_clean.csv", index=False, encoding="utf-8-sig")
print("الوطني بعد:", n.shape)


# ============ 3) ملف السكان ============
p = pd.read_csv(RAW / "popolazione.csv", dtype={"codice_regione": "string"})
print("السكان قبل:", p.shape)

for c in ["codice_nuts_1", "descrizione_nuts_1", "codice_nuts_2",
          "denominazione_regione", "sigla_regione", "range_eta"]:
    p[c] = clean_text(p[c])
p["codice_regione"] = p["codice_regione"].str.zfill(2)
p["denominazione_regione"] = p["denominazione_regione"].replace(NAME_FIX)
p["latitudine_regione"] = p["latitudine_regione"].round(8)
p["longitudine_regione"] = p["longitudine_regione"].round(8)
p = to_int(p, ["totale_genere_maschile", "totale_genere_femminile", "totale_generale"])

# لومبارديا فئاتها العمرية مختلفة (05-11 و12-19 بدل 0-15 و16-19)
# يعني مفيهاش الأطفال من 0 لـ 4 سنين، فنعمل علامة عليها
standard = p.groupby("codice_regione")["range_eta"].transform(lambda s: "0-15" in set(s))
p["flag_age_bands_incomplete"] = ~standard

p = p.sort_values(["codice_regione", "range_eta"]).reset_index(drop=True)
p.to_csv(CLEAN / "popolazione_clean.csv", index=False, encoding="utf-8-sig")
print("السكان بعد:", p.shape)

# جدول الأقاليم (21 صف): معلومات الإقليم + إجمالي السكان
lookup = (p.groupby(["codice_regione", "denominazione_regione", "sigla_regione",
                     "codice_nuts_1", "descrizione_nuts_1", "codice_nuts_2",
                     "latitudine_regione", "longitudine_regione"], as_index=False)
            .agg(population_total=("totale_generale", "sum"),
                 flag_age_bands_incomplete=("flag_age_bands_incomplete", "max")))
lookup["population_total"] = lookup["population_total"].astype("Int64")
lookup.to_csv(CLEAN / "regioni_lookup.csv", index=False, encoding="utf-8-sig")
print("جدول الأقاليم:", lookup.shape)


# ============ 4) ملف الملاحظات ============
t = pd.read_csv(RAW / "note.csv")
print("الملاحظات قبل:", t.shape)

t["date"] = pd.to_datetime(t["data"]).dt.strftime("%Y-%m-%d")
t = t.drop(columns="data")
# نظّف التابات والمسافات (النص الإيطالي زي ما هو)
t["note"] = clean_text(t["note"])
t["has_note"] = t["note"].notna()
t = t.sort_values("date").reset_index(drop=True)
t = t[["date", "note", "has_note"]]
t.to_csv(CLEAN / "note_clean.csv", index=False, encoding="utf-8-sig")
print("الملاحظات بعد:", t.shape, "| فيها نص:", int(t["has_note"].sum()))

print("\nخلصنا. الملفات في فولدر clean")
