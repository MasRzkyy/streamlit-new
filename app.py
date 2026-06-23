import streamlit as st 
import numpy as np 
import joblib 
import matplotlib.pyplot as plt 
import pandas as pd

# Konfigurasi Halaman (Centered Layout)
st.set_page_config(
    page_title="Proyeksi Bisnis",
    page_icon="🚀",
    layout="centered" # Menggunakan centered untuk gaya "card" terfokus
)

# Custom CSS bergaya modern minimalis dengan gradient text
st.markdown("""
<style>
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

# Header Centered
st.markdown('<div class="title-text">Kalkulator Profit Pintar</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Gunakan AI untuk mengoptimalkan strategi marketing dan diskon Anda.</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model 'model_keuntungan.pkl' tidak ditemukan di direktori ini!")
    st.stop()

# --- BAGIAN INPUT TANPA SIDEBAR ---
# Kita gunakan box (container) untuk input agar rapi
st.markdown("### ⚙️ Parameter Kebijakan")
st.write("Masukkan nilai anggaran dan diskon di bawah ini:")

# Menggunakan kolom untuk input yang sejajar dan tipe number_input
col_input1, col_input2 = st.columns(2)

with col_input1:
    # Menggunakan slider dengan format custom agar lebih visual
    iklan = st.slider("💰 Budget Iklan", min_value=0, max_value=50, value=10, step=1, format="Rp %d Jt")
with col_input2:
    # Menggunakan radio buttons horizontal untuk pilihan diskon yang cepat
    diskon = st.radio("🏷️ Persentase Diskon", options=[0, 5, 10, 15, 20], index=1, horizontal=True, format_func=lambda x: f"{x}%")

# --- PREDIKSI ---
input_data = np.array([[iklan, diskon]])
prediksi = model.predict(input_data)[0]
baseline = 100.0
selisih = prediksi - baseline

st.markdown("---")

# --- MENGGUNAKAN TABS UNTUK UI BERBEDA ---
tab1, tab2 = st.tabs(["🎯 Hasil Simulasi", "📊 Analisis Sensitivitas"])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.info("💡 **Insight AI:** Berdasarkan model prediktif, ini adalah estimasi dari kombinasi kebijakan Anda.")
        st.metric(
            label="Estimasi Total Profit", 
            value=f"Rp {prediksi:,.1f} Jt", 
            delta=f"{'+' if selisih>=0 else ''}Rp {selisih:,.1f} Jt dari Baseline"
        )
        st.caption(f"*Baseline profit ditetapkan pada Rp {baseline} Juta.")

    with res_col2:
        # Visualisasi dengan Donut Chart (Pie chart berlubang)
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # Validasi jika prediksi negatif (walau jarang dalam skenario simpel)
        p_val = max(0, prediksi) 
        
        sizes = [baseline, p_val]
        labels = ['Baseline', 'Proyeksi Baru']
        colors = ['#E5E7EB', '#00f2fe']
        explode = (0, 0.05) # Pisahkan sedikit
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=False, startangle=140, textprops={'fontsize': 10, 'fontweight': 'bold'})
        
        # Gambar lingkaran di tengah untuk efek Donut
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)
        
        ax.axis('equal') 
        plt.tight_layout()
        
        # Menampilkan plot di streamlit agar ukurannya pas
        st.pyplot(fig, width='stretch')

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Bagaimana jika Anda mengubah budget iklan?")
    st.write(f"Grafik dan tabel di bawah menunjukkan proyeksi profit jika Anda menaikkan/menurunkan budget iklan dengan **diskon tetap** di angka {diskon}%.")
    
    # Generate simulasi otomatis untuk berbagai nilai iklan
    # Memastikan nilai input pengguna saat ini selalu ada dalam tabel
    skenario_iklan = sorted(list(set([0, 10, 20, 30, 40, 50, iklan])))
    
    hasil_skenario = []
    
    for i in skenario_iklan:
        p = model.predict(np.array([[i, diskon]]))[0]
        hasil_skenario.append(p)
        
    df_sensitivitas = pd.DataFrame({
        "Anggaran Iklan (Juta Rp)": skenario_iklan,
        "Proyeksi Profit (Juta Rp)": hasil_skenario
    })
    
    # Highlight baris jika nilainya sama dengan input saat ini
    def highlight_current(row):
        if row['Anggaran Iklan (Juta Rp)'] == iklan:
            return ['background-color: #DBEAFE; color: #1E3A8A; font-weight: bold'] * len(row)
        return [''] * len(row)
        
    col_tab1, col_tab2 = st.columns([1, 1.5])
    
    with col_tab1:
        # Tampilkan tabel dengan format 2 desimal di belakang koma
        st.dataframe(
            df_sensitivitas.style.apply(highlight_current, axis=1).format({"Proyeksi Profit (Juta Rp)": "{:.2f}"}), 
            hide_index=True
        )
        
    with col_tab2:
        # Line chart bawaan Streamlit
        st.line_chart(df_sensitivitas.set_index("Anggaran Iklan (Juta Rp)"), color="#00f2fe")