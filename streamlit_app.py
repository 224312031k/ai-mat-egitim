import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Matematik Eğitimi", layout="wide")

st.title("📊 Yapay Zeka Destekli Bireysel Matematik Eğitimi")

st.write("""
Bu uygulama, öğrencilerin anket verilerine dayalı öğrenme stillerini belirler
ve buna uygun bireysel öneriler sunar.
""")

uploaded_file = st.file_uploader("📂 Anket verilerini yükleyin (CSV formatında)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Veri başarıyla yüklendi!")
    st.dataframe(df.head())
else:
    st.info("Devam etmek için lütfen bir CSV dosyası yükleyin.")
