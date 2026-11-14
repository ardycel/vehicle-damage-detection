import streamlit as st
from PIL import Image
from src.detector import DamageDetector

st.set_page_config(page_title="AI Hasar Tespit Sistemi", layout="wide")
st.title("🔍 Görüntü Tabanlı Yüzey / Araç Hasar Tespit Sistemi")
st.write("Hasar tespiti yapmak istediğiniz fotoğrafı yükleyin.")

@st.cache_resource
def load_detector():
    return DamageDetector("models/best.pt")

detector = load_detector()

# Yan menü ayarları
conf_threshold = st.sidebar.slider("Güven Eşiği (Confidence Threshold)", 0.10, 0.90, 0.30, 0.05)

uploaded_file = st.file_uploader("Görsel Seçin (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Orijinal Görsel")
        st.image(image, use_container_width=True)

    with st.spinner("Model görüntüyü analiz ediyor..."):
        result_img, detections = detector.predict(image, conf_threshold)

    with col2:
        st.subheader("Tespit Sonucu")
        st.image(result_img, use_container_width=True)

    st.divider()
    st.subheader("Tespit Özeti")
    if detections:
        for idx, item in enumerate(detections, 1):
            st.success(f"**{idx}. Tespit:** {item['label']} — Güven Skoru: **%{item['confidence']}**")
    else:
        st.info("Seçilen güven eşiğinde herhangi bir hasar tespit edilmedi.")