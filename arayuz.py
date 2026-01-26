import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt

# --- GÖRSEL TEMA AYARLARI (Custom CSS) ---
st.set_page_config(page_title="Biometric Insights Lab", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stAlert {
        border-radius: 10px;
    }
    div.stButton > button:first-child {
        background-color: #2e4053;
        color: white;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BAŞLIK ALANI ---
col_head1, col_head2 = st.columns([1, 4])
with col_head1:
    st.write("") # Logo alanı olarak boş bırakıldı
with col_head2:
    st.title("🔬 Biyometrik Veri Analiz Platformu")
    st.caption("Gelişmiş Görüntü İşleme ve Karakter Tanıma Terminali")

# --- MODÜLLERİN YÜKLENMESİ (Mantık Değişmedi) ---
@st.cache_resource
def load_ocr_model():
    try:
        return joblib.load("ocr_config.pkl")
    except:
        return None

model = load_ocr_model()

# --- MATEMATİKSEL FONKSİYONLAR (Dokunulmadı) ---
def extract_minutiae(skeleton):
    """İskelet üzerindeki T (Terminal) ve I (Intersection) noktalarını bulur"""
    terminals = []
    intersections = []
    h, w = skeleton.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if skeleton[i, j] == 255:
                neighbor_sum = np.sum(skeleton[i-1:i+2, j-1:j+2]) / 255 - 1
                if neighbor_sum == 1:
                    terminals.append((j, i))
                elif neighbor_sum >= 3:
                    intersections.append((j, i))
    return terminals, intersections

# --- YAN PANEL (REORGANİZE EDİLDİ) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fingerprint-scanning.png", width=80)
    st.header("Sistem Ayarları")
    
    app_mode = st.selectbox("Çalışma Modu Seçiniz:", 
                            ["Harf Tanıma (OCR)", "Yüz Tanıma (Geometrik)", "Avuç İçi (Yapısal)", "Parmak İzi (Korelasyon)"])
    
    st.divider()
    
    with st.expander("Görüntü Parametreleri", expanded=True):
        threshold_val = st.slider("Binarizasyon Eşiği:", 0, 255, 127)
    
    uploaded_file = st.file_uploader("Veri Kaynağı Yükle (.png, .jpg)", type=["png", "jpg", "jpeg"])

# --- ANA PANEL ---
if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # Tasarım değişikliği: Yan yana sütunlar yerine, geniş bir işlem alanı
    main_container = st.container()
    
    with main_container:
        # Alt Bilgi ve Akademik Notlar Expander'a taşındı (Tasarım farkı)
        with st.expander("📝 Metodolojik Altyapı ve Akademik Notlar", expanded=False):
            if app_mode == "Yüz Tanıma (Geometrik)":
                st.write("- **Ağırlık Merkezi:** Siyah piksellerin koordinat toplamının toplam sayıya oranı.")
                st.write("- **Normalizasyon:** Farklı ölçekteki yüzleri tanımak için en/boy oranlaması.")
            elif app_mode == "Avuç İçi (Yapısal)":
                st.write("- **Stentiford Algoritması:** Şablon tabanlı 'İşaretle ve Yok Et' mantığı.")
                st.write("- **Bağlantı Sayısı ($C_n$):** Bir pikselin silinip silinmeyeceğine karar veren matematiksel kural.")

        st.divider()
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("### Giriş Verisi")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with c2:
            st.markdown(f"### {app_mode} Analizi")
            
            if app_mode == "Harf Tanıma (OCR)":
                if model:
                    res = cv2.resize(binary, (28, 28))
                    feat = hog(res, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
                    tahmin = model.predict([feat])[0]
                    st.success(f"**Sonuç:** {tahmin}")
                    st.progress(model.predict_proba([feat]).max(), text=f"Güven Skoru: %{model.predict_proba([feat]).max()*100:.2f}")
                else:
                    st.warning("OCR Modeli (ocr_config.pkl) yüklenemedi.")

            elif app_mode == "Yüz Tanıma (Geometrik)":
                M = cv2.moments(binary)
                if M["m00"] != 0:
                    gx, gy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    h, w = binary.shape
                    norm_x, norm_y = gx/w, gy/h
                    
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Normalize Gx'", round(norm_x, 3))
                    m_col2.metric("Normalize Gy'", round(norm_y, 3))
                    
                    cv2.circle(img, (gx, gy), 15, (255, 0, 0), -1)
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Hesaplanan Geometrik Merkez")

            elif app_mode == "Avuç İçi (Yapısal)":
                ske = cv2.ximgproc.thinning(binary)
                terminals, intersections = extract_minutiae(ske)
                
                vis = cv2.cvtColor(ske, cv2.COLOR_GRAY2RGB)
                for pt in terminals: cv2.circle(vis, pt, 3, (255, 0, 0), -1)
                for pt in intersections: cv2.circle(vis, pt, 3, (0, 255, 0), -1)
                
                st.image(vis, caption="Minutiae (Uç/Kesişim) Haritası", use_container_width=True)
                st.info(f"Sistem toplam {len(terminals)} adet uç nokta doğruladı.")

else:
    st.empty()
    col_info1, col_info2, col_info3 = st.columns([1,2,1])
    with col_info2:
        st.info("👋 Hoş Geldiniz. Analiz için sol menüden görüntü yükleyiniz.")