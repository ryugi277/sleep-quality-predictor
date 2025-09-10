import json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/model.pkl")
META_PATH  = Path("models/meta.json")

st.set_page_config(page_title="Sleep Quality Predictor", page_icon="😴", layout="wide")

# ---------- Header ----------
st.markdown("""
<h1 style="margin-bottom:0">😴 Sleep Quality Predictor</h1>
<p style="color:#6b7280; font-size:1.05rem; margin-top:.25rem">
Isi form singkat di bawah untuk memperkirakan <b>kualitas tidur</b> Anda (Baik/Buruk) beserta skor probabilitas dan saran praktis.
</p>
<hr style="margin: .75rem 0 1.25rem 0">
""", unsafe_allow_html=True)

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta

model, meta = load_artifacts()
num_cols = meta["num_cols"]
cat_cols = meta["cat_cols"]
all_cols = num_cols + cat_cols

# ---------- Form ----------
st.subheader("Form Input Gaya Hidup & Tidur")

col1, col2 = st.columns(2, gap="large")

with col1:
    sleep_duration = st.number_input("Durasi tidur (jam/malam)", 0.0, 15.0, 6.5, step=0.1)
    physical = st.number_input("Aktivitas fisik (menit/hari)", 0, 300, 45, step=5)
    stress = st.slider("Tingkat stres (1 rendah – 10 tinggi)", 1, 10, 5)
    age = st.number_input("Usia", 10, 90, 28)

with col2:
    heart = st.number_input("Detak jantung istirahat (bpm)", 40, 200, 72)
    steps = st.number_input("Langkah harian (steps)", 0, 30000, 7000, step=500)
    gender = st.selectbox("Gender", ["Male","Female"])
    occupation = st.selectbox(
        "Pekerjaan",
        ["Software Engineer","Doctor","Nurse","Teacher","Sales Representative","Accountant","Other"]
    )
    bmi_cat = st.selectbox("Kategori BMI", ["Underweight","Normal","Overweight","Obese"])
    sleep_dis = st.selectbox("Gangguan tidur", ["None","Insomnia","Sleep Apnea"])

with st.container():
    cols = st.columns([1,1,1,2])
    with cols[0]:
        predict_click = st.button("🔮 Prediksi", use_container_width=True)
    with cols[1]:
        reset_click = st.button("↺ Reset", use_container_width=True)
    with cols[2]:
        th = st.slider("Threshold 'Baik' (%)", 40, 80, 50, help="Di atas ini dihitung 'Baik'")

if reset_click:
    st.experimental_rerun()

# ---------- Prediction ----------
if predict_click:
    row = {
        "sleep_duration": sleep_duration,
        "physical_activity_level": physical,
        "stress_level": stress,
        "age": age,
        "heart_rate": heart,
        "daily_steps": steps,
        "gender": gender,
        "occupation": occupation,
        "bmi_category": bmi_cat,
        "sleep_disorder": None if sleep_dis=="None" else sleep_dis,
    }
    # pastikan urutan kolom sesuai pipeline
    row = {k: row.get(k, np.nan) for k in all_cols}
    X = pd.DataFrame([row], columns=all_cols)

    prob_good = float(model.predict_proba(X)[0,1])
    label = "Baik" if (prob_good*100) >= th else "Buruk"

    st.write("---")
    # KPI cards
    k1, k2, k3 = st.columns(3)
    k1.metric("Prediksi", label)
    k2.metric("Probabilitas 'Baik'", f"{prob_good*100:.1f}%")
    k3.metric("Durasi tidur", f"{sleep_duration:.1f} jam")

    # progress bar skor
    st.caption("Skor kualitas tidur (semakin kanan semakin baik)")
    st.progress(min(max(prob_good, 0.0), 1.0))

    # rekomendasi berbasis aturan sederhana
    tips = []
    if sleep_duration < 6:
        tips.append("Tambah durasi tidur (target 7–9 jam untuk dewasa).")
    if stress >= 7:
        tips.append("Kelola stres: atur jadwal, teknik pernapasan, journaling.")
    if physical < 30:
        tips.append("Tingkatkan aktivitas fisik ≥30 menit/hari (jalan cepat).")
    if heart > 85:
        tips.append("Periksa kebugaran & konsumsi kafein menjelang tidur.")
    if steps < 5000:
        tips.append("Naikkan langkah harian (≥7.000) untuk kualitas tidur lebih baik.")
    if sleep_dis != "None":
        tips.append("Konsultasi profesional untuk gangguan tidur yang terindikasi.")

    if tips:
        st.info("**Saran cepat untuk meningkatkan kualitas tidur:**\n- " + "\n- ".join(tips))
    else:
        st.success("Bagus! Kebiasaan Anda sudah mendukung kualitas tidur yang baik.")

    # Feature importance (global)
    try:
        importances = model.named_steps["model"].feature_importances_
        ohe = model.named_steps["prep"].named_transformers_["cat"]
        num_names = num_cols
        cat_names = list(ohe.get_feature_names_out(cat_cols))
        feat_names = num_names + cat_names
        imp_df = (pd.DataFrame({"Fitur": feat_names, "Penting": importances})
                  .sort_values("Penting", ascending=False).head(10))
        with st.expander("🔎 Fitur paling berpengaruh (global)"):
            st.dataframe(imp_df, use_container_width=True)
    except Exception:
        st.caption("Feature importance tidak tersedia untuk konfigurasi saat ini.")

# ---------- Help / About ----------
with st.expander("ℹ️ Tentang Model & Batasan"):
    st.markdown(f"""
    - Model: RandomForestClassifier dalam pipeline preprocessing (standarisasi + one-hot).
    - Akurasi validasi: **{meta.get('accuracy', 0):.3f}** (berdasarkan dataset publik).
    - Target: *Baik* jika **quality_of_sleep ≥ 7** pada data pelatihan.
    - Ini alat **edukasi**, bukan perangkat diagnosis medis.
    """)

st.write("---")
st.caption("© 2025 Sleep App Demo • Developed by Rina Yulius and Team.")
