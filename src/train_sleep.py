import json, joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = Path("data/datasleep.csv")
MODEL_PATH = Path("models/model.pkl")
META_PATH  = Path("models/meta.json")

# 1) load data
df = pd.read_csv(DATA_PATH)

# Normalisasi nama kolom agar aman (beberapa dataset pakai spasi)
df.columns = [c.strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]

# Map kolom yang biasanya ada di dataset ini
# contoh: person id, gender, age, occupation, sleep_duration, quality_of_sleep,
# physical_activity_level, stress_level, bmi_category, blood_pressure, heart_rate,
# daily_steps, sleep_disorder
target_col = "quality_of_sleep"  # skala 1..10. Kita ubah ke klasifikasi: baik (>=7) vs buruk (<7)
if target_col not in df.columns:
    raise ValueError(f"Kolom '{target_col}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")

df = df.dropna(subset=[target_col]).copy()
df["sleep_quality_label"] = (df[target_col] >= 7).astype(int)  # 1=Baik, 0=Burok

# fitur yang akan dipakai
num_cols = [c for c in ["sleep_duration","physical_activity_level","stress_level",
                        "age","heart_rate","daily_steps"] if c in df.columns]
cat_cols = [c for c in ["gender","occupation","bmi_category","sleep_disorder"] if c in df.columns]

X = df[num_cols + cat_cols].copy()
y = df["sleep_quality_label"].copy()

# 2) split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) preprocessing
ct = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# 4) model
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

pipe = Pipeline(steps=[("prep", ct), ("model", clf)])

# 5) train
pipe.fit(X_train, y_train)

# 6) eval
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", round(acc,3))
print(classification_report(y_test, y_pred, digits=3))

# 7) save
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, MODEL_PATH)

META_PATH.write_text(
    json.dumps({
        "target_definition": "sleep_quality_label = 1 if quality_of_sleep >= 7 else 0",
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "accuracy": acc
    }, indent=2),
    encoding="utf-8"
)
print(f"Saved -> {MODEL_PATH}")
