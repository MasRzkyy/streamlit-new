import streamlit as st 
import numpy as np 
import joblib 
import matplotlib.pyplot as plt 
import pandas as pd

st.set_page_config(
    page_title="Streamlit",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .title-text {
        text-align: center;
        font-size: 3rem;
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle-text {
        text-align: center;
        color: #6B7280;
        margin-bottom: 3rem;
        font-size: 1.2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load('model_keuntungan.pkl')
    except Exception:
        return None

model = load_model()

st.markdown('<div class="title-text">Prediksi Keuntungan</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Prepared by : Mohammad Rizky Andi Putra</div>', unsafe_allow_html=True)

if model is None:
    st.error("Model 'model_keuntungan.pkl' tidak ditemukan di direktori ini!")
    st.stop()

st.markdown("### Input Kebijakan")
st.write("Masukkan nilai anggaran dan diskon di bawah ini:")

col_input1, col_input2 = st.columns(2)

with col_input1:
    iklan = st.slider("Budget Iklan", min_value=0, max_value=50, value=10, step=1, format="Rp %d Jt")
with col_input2:
    diskon = st.radio("Persentase Diskon", options=[0, 5, 10, 15, 20], index=1, horizontal=True, format_func=lambda x: f"{x}%")

# --- PREDIKSI ---
input_data = np.array([[iklan, diskon]])
prediksi = model.predict(input_data)[0]
baseline = 100.0
selisih = prediksi - baseline

st.markdown("---")

st.markdown("### Hasil Simulasi")
st.markdown("<br>", unsafe_allow_html=True)
res_col1, res_col2 = st.columns([1, 1])

with res_col1:
    st.metric(
        label="Estimasi Total Profit", 
        value=f"Rp {prediksi:,.1f} Jt", 
        delta=f"{'+' if selisih>=0 else ''}Rp {selisih:,.1f} Jt dari Baseline"
    )
    st.caption(f"*Baseline profit ditetapkan pada Rp {baseline} Juta.")

with res_col2:
    fig, ax = plt.subplots(figsize=(4, 4))
    p_val = max(0, prediksi) 
    
    sizes = [baseline, p_val]
    labels = ['Baseline', 'Proyeksi Baru']
    colors = ['#E5E7EB', '#00f2fe']
    explode = (0, 0.05) 
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=False, startangle=140, radius=1, textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal') 
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    st.pyplot(fig, width='stretch')