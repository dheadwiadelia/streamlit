import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load dataset
df = pd.read_csv("Diabetes_dataset.csv")

# LabelEncoder terpisah untuk setiap kolom kategorik
le_activity = LabelEncoder()
le_comorb = LabelEncoder()

# Encode kolom kategorik
df['Physical_Activity_Level'] = le_activity.fit_transform(df['Physical_Activity_Level'])
df['Comorbidities'] = le_comorb.fit_transform(df['Comorbidities'].astype(str))

# Bersihkan data
df.fillna(0, inplace=True)

# Streamlit App
st.title("Prediksi Risiko Diabetes Berdasarkan Gaya Hidup dan Faktor Klinis")

# Input pengguna
st.header("Masukkan Data Pasien")
age = st.number_input("Usia", min_value=10, max_value=100, value=30)
bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=10.0, max_value=50.0, value=22.5)
diet = st.slider("Skor Pola Makan (1–10)", 1, 10, 5)
activity = st.selectbox("Tingkat Aktivitas Fisik", ['Low', 'Moderate', 'High'])

# Ambil label comorbid berdasarkan data
comorb_labels = le_comorb.inverse_transform(np.unique(df['Comorbidities']))
comorb = st.selectbox("Penyakit Penyerta", comorb_labels)

# Encode input pengguna
activity_encoded = le_activity.transform([activity])[0]
comorb_encoded = le_comorb.transform([comorb])[0]

# Siapkan data training
features = ['Age', 'BMI', 'Diet_Score', 'Physical_Activity_Level', 'Comorbidities']
X = df[features]
y = df['Diabetes']

# Train model regresi logistik
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Prediksi berdasarkan input pengguna
user_input = np.array([[age, bmi, diet, activity_encoded, comorb_encoded]])
pred = model.predict(user_input)
prob = model.predict_proba(user_input)[0][1]

# Tampilkan hasil prediksi
st.header("Hasil Prediksi")
if pred[0] == 1:
    st.error(f"⚠️ Pasien diprediksi **BERISIKO** diabetes dengan probabilitas {prob:.2f}")
else:
    st.success(f"✅ Pasien diprediksi **TIDAK BERISIKO** diabetes dengan probabilitas {1 - prob:.2f}")
