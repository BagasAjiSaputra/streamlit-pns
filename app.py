import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Konfigurasi Halaman (Harus dipanggil pertama kali sebelum elemen Streamlit lain)
st.set_page_config(
    page_title="Dashboard Prediksi Keuntungan",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Memuat Model & Data Historis
@st.cache_resource
def load_model():
    return joblib.load('model_fix.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Data historis (sesuai data latih di Project14.ipynb)
X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
y_train = np.array([50, 80, 110, 90, 150])

# Menghitung baseline prediction (Iklan = 10 Juta, Diskon = 10%) secara dinamis dari model
baseline_input = np.array([[10, 10]])
baseline_pred = model.predict(baseline_input)[0]

# 2. Injeksi CSS Custom untuk Tampilan Neobrutalism
st.markdown("""
<style>
    /* Styling font global */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Judul Utama */
    .main-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 900;
        font-size: 2.5rem;
        color: #000000;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: -1.5px;
        background-color: #FDE047; /* Bright Yellow */
        border: 4px solid #000000;
        padding: 15px 30px;
        box-shadow: 6px 6px 0px 0px #000000;
        display: block;
        margin: 10px auto 10px auto;
        box-sizing: border-box;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-color);
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 35px;
    }
    
    /* Card KPI */
    .kpi-card {
        background-color: #FFFFFF;
        border: 4px solid #000000;
        border-radius: 0px;
        padding: 24px;
        box-shadow: 6px 6px 0px 0px #000000;
        margin-bottom: 25px;
        color: #000000;
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    
    .kpi-card:hover {
        transform: translate(-3px, -3px);
        box-shadow: 9px 9px 0px 0px #000000;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 800;
        color: #000000;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #000000;
        padding-bottom: 5px;
        margin-bottom: 12px;
    }
    
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 900;
        color: #000000;
        margin: 10px 0;
    }
    
    /* Neobrutalist Badges */
    .delta-badge {
        display: inline-flex;
        align-items: center;
        font-size: 0.85rem;
        font-weight: 800;
        padding: 6px 14px;
        border: 2px solid #000000;
        border-radius: 0px;
        margin-top: 8px;
        text-transform: uppercase;
        box-shadow: 2px 2px 0px #000000;
    }
    
    .delta-positive {
        background-color: #4ADE80; /* Bright Green */
        color: #000000;
    }
    
    .delta-negative {
        background-color: #F87171; /* Bright Red */
        color: #000000;
    }
    
    /* Highlight model metadata info */
    .meta-info-box {
        font-size: 0.8rem;
        font-weight: 700;
        padding: 8px 12px;
        background-color: #F1F5F9;
        border: 2px solid #000000;
        border-radius: 0px;
        margin-top: 15px;
        color: #000000;
        text-transform: uppercase;
        box-shadow: 2px 2px 0px #000000;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar - Profile & Input
# Header Profil Neobrutalist
st.sidebar.markdown(f"""
<div style="background-color: #FFFFFF; border: 3px solid #000000; border-radius: 0px; padding: 20px; text-align: center; margin-bottom: 25px; box-shadow: 5px 5px 0px 0px #000000; color: #000000;">
    <div style="width: 55px; height: 55px; background-color: #60A5FA; color: #000000; border: 3px solid #000000; border-radius: 0px; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px auto; font-size: 1.3rem; font-weight: 900; box-shadow: 3px 3px 0px #000000;">
        FA
    </div>
    <h4 style="margin: 0; font-weight: 900; font-size: 1.05rem; color: #000000; text-transform: uppercase;">Bagas Aji Saputra</h4>
    <p style="margin: 3px 0 0 0; font-size: 0.8rem; color: #000000; font-weight: 700; opacity: 0.7;">NPM 2313020028</p>
    <div style="margin-top: 12px; display: inline-block; font-size: 0.75rem; color: #000000; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; background-color: #FCA5A5; border: 2px solid #000000; padding: 3px 10px; box-shadow: 2px 2px 0px #000000;">
        PNS Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Input Kebijakan")
st.sidebar.markdown("<p style='font-size:0.85rem; opacity:0.8; margin-top:-10px; font-weight:600;'>Sesuaikan parameter untuk memicu prediksi keuntungan baru.</p>", unsafe_allow_html=True)

# Definisikan state untuk reset
if "iklan_val" not in st.session_state:
    st.session_state.iklan_val = 10
if "diskon_val" not in st.session_state:
    st.session_state.diskon_val = 10

def reset_to_baseline():
    st.session_state.iklan_val = 10
    st.session_state.diskon_val = 10

# Slider input
iklan = st.sidebar.slider("Biaya Iklan (Juta)", min_value=0, max_value=50, key="iklan_val")
diskon = st.sidebar.slider("Diskon (%)", min_value=0, max_value=20, key="diskon_val")

# Tombol reset dengan style neobrutalist yang didukung HTML via sidebar
# Namun karena Streamlit button standard, kita gunakan parameter native Streamlit
st.sidebar.button("Reset ke Baseline", on_click=reset_to_baseline, use_container_width=True)

# 4. Prediksi (What-If Analysis)
input_data = np.array([[iklan, diskon]])
prediksi = model.predict(input_data)[0]

# Hitung selisih dibandingkan baseline yang dihitung model
delta = prediksi - baseline_pred
delta_pct = (delta / baseline_pred) * 100 if baseline_pred != 0 else 0

# Definisikan warna & icon berdasarkan arah delta
if delta >= 0:
    delta_class = "delta-positive"
    delta_text = "+"
    delta_pct_str = f"+{delta_pct:.2f}%"
    chart_highlight_color = "#4ADE80" # Neobrutalist Green
else:
    delta_class = "delta-negative"
    delta_text = ""
    delta_pct_str = f"{delta_pct:.2f}%"
    chart_highlight_color = "#F87171" # Neobrutalist Red

# 5. Tampilan Main Panel
st.markdown('<h1 class="main-title">DASHBOARD SIMULASI KEUNTUNGAN</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Platform Simulasi Prediktif untuk Pengambilan Keputusan Bisnis Secara Pintar</p>', unsafe_allow_html=True)

# Tata Letak Grid: Tiga Card KPI Kolom di Bagian Atas
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Skenario Baru (Prediksi)</div>
        <div class="kpi-value" style="color: #2563EB;">Rp {prediksi:.2f} Juta</div>
        <div class="meta-info-box">
            Iklan: {iklan} Jt • Diskon: {diskon}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Baseline (Kondisi Awal)</div>
        <div class="kpi-value" style="color: #000000; opacity: 0.85;">Rp {baseline_pred:.2f} Juta</div>
        <div class="meta-info-box" style="background-color: #F1F5F9;">
            Iklan: 10 Jt • Diskon: 10%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Estimasi Dampak Profit</div>
        <div class="kpi-value" style="font-size: 2.1rem; font-weight: 900; color: #000000; margin: 10px 0;">
            Rp {delta_text}{delta:.2f} Juta
        </div>
        <span class="delta-badge {delta_class}">{delta_pct_str} dibanding baseline</span>
    </div>
    """, unsafe_allow_html=True)

# Pembatas Konten
st.markdown("<br>", unsafe_allow_html=True)

# Grid Visualisasi & Detail Insight
col_viz, col_insight = st.columns([1.6, 1])

with col_viz:
    st.markdown("### Visualisasi & Analisis Tren")
    
    # Penggunaan Tab agar visualisasi bersih dan terstruktur
    tab1, tab2 = st.tabs(["Analisis Sensitivitas & Historis", "Perbandingan Skenario"])
    
    with tab1:
        # Membuat visualisasi trend 2D berkualitas tinggi neobrutalist
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
        fig.patch.set_facecolor('none')  # Background transparan agar menyatu dengan Streamlit
        
        # --- Subplot 1: Keuntungan vs Biaya Iklan ---
        ax1.set_facecolor('#FFFFFF')
        # Line prediksi model (diskon tetap sesuai input slider)
        iklan_range = np.linspace(0, 50, 100)
        pred_line_iklan = model.predict(np.column_stack((iklan_range, np.full_like(iklan_range, diskon))))
        ax1.plot(iklan_range, pred_line_iklan, color='#2563EB', linewidth=3, label='Garis Regresi')
        
        # Scatter data training asli (Neobrutalist styling dengan border hitam tebal)
        ax1.scatter(X_train[:, 0], y_train, color='#E2E8F0', edgecolors='#000000', linewidths=2, s=60, alpha=1.0, zorder=3, label='Data Historis')
        
        # Sorot titik pilihan user saat ini (Bintang besar dengan outline hitam)
        ax1.scatter(iklan, prediksi, color=chart_highlight_color, edgecolors='#000000', linewidths=2.5, s=250, marker='*', zorder=5, label='Skenario Anda')
        
        ax1.set_title("Keuntungan vs Biaya Iklan\n(Diskon Tetap)", fontsize=9, fontweight='black', color='#000000')
        ax1.set_xlabel("Biaya Iklan (Juta)", fontsize=8, fontweight='bold', color='#000000')
        ax1.set_ylabel("Keuntungan (Juta)", fontsize=8, fontweight='bold', color='#000000')
        ax1.grid(True, linestyle='-', alpha=0.15, color='#000000')
        ax1.tick_params(labelsize=8, colors='#000000')
        
        # Spines hitam tebal neobrutal
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_color('#000000')
            spine.set_linewidth(2.5)
        
        # --- Subplot 2: Keuntungan vs Diskon ---
        ax2.set_facecolor('#FFFFFF')
        # Line prediksi model (iklan tetap sesuai input slider)
        diskon_range = np.linspace(0, 20, 100)
        pred_line_diskon = model.predict(np.column_stack((np.full_like(diskon_range, iklan), diskon_range)))
        ax2.plot(diskon_range, pred_line_diskon, color='#2563EB', linewidth=3)
        
        # Scatter data training asli
        ax2.scatter(X_train[:, 1], y_train, color='#E2E8F0', edgecolors='#000000', linewidths=2, s=60, alpha=1.0, zorder=3)
        
        # Sorot titik pilihan user saat ini
        ax2.scatter(diskon, prediksi, color=chart_highlight_color, edgecolors='#000000', linewidths=2.5, s=250, marker='*', zorder=5)
        
        ax2.set_title("Keuntungan vs Diskon\n(Iklan Tetap)", fontsize=9, fontweight='black', color='#000000')
        ax2.set_xlabel("Diskon (%)", fontsize=8, fontweight='bold', color='#000000')
        ax2.set_ylabel("Keuntungan (Juta)", fontsize=8, fontweight='bold', color='#000000')
        ax2.grid(True, linestyle='-', alpha=0.15, color='#000000')
        ax2.tick_params(labelsize=8, colors='#000000')
        
        # Spines hitam tebal neobrutal
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_color('#000000')
            spine.set_linewidth(2.5)
        
        # Handler legend satu untuk keseluruhan chart
        fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', ncol=3, frameon=True, edgecolor='#000000', facecolor='#FFFFFF', fontsize=8)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        st.pyplot(fig)
        
    with tab2:
        # Grafik Komparasi Baseline vs Skenario Baru
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3), dpi=150)
        fig_bar.patch.set_facecolor('none')
        ax_bar.set_facecolor('#FFFFFF')
        
        # Bar horizontal neobrutalist dengan border hitam tebal
        bars = ax_bar.barh(['Baseline', 'Skenario Baru'], [baseline_pred, prediksi], height=0.5, color=['#E2E8F0', '#FDE047'], edgecolor='#000000', linewidth=2.5)
        
        # Custom styling bar chart horizontal neobrutal
        for spine in ax_bar.spines.values():
            spine.set_visible(True)
            spine.set_color('#000000')
            spine.set_linewidth(2.5)
            
        ax_bar.set_xlabel("Keuntungan (Juta Rupiah)", fontsize=8, fontweight='bold', color='#000000')
        ax_bar.tick_params(labelsize=8, colors='#000000')
        ax_bar.grid(axis='x', linestyle='-', alpha=0.15, color='#000000')
        
        # Menambahkan label teks di dalam/ujung bar
        for bar in bars:
            width = bar.get_width()
            ax_bar.text(width + 2, bar.get_y() + bar.get_height()/2, f'Rp {width:.2f} Jt', 
                        ha='left', va='center', fontweight='black', fontsize=8, color='#000000')
            
        plt.tight_layout()
        st.pyplot(fig_bar)

with col_insight:
    st.markdown("### Wawasan & Analisis Model")
    
    # Menampilkan interpretasi model regresi linier secara dinamis
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        coef_iklan = model.coef_[0]
        coef_diskon = model.coef_[1]
        
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 4px solid #000000; border-radius: 0px; padding: 24px; box-shadow: 6px 6px 0px 0px #000000; color: #000000;">
            <h5 style="margin-top: 0; color: #000000; font-weight: 900; text-transform: uppercase; border-bottom: 2px solid #000000; padding-bottom: 5px; margin-bottom: 12px;">Karakteristik Model</h5>
            <p style="font-size: 0.85rem; line-height: 1.5; color: #000000; font-weight: 700;">
                Persamaan Regresi Linier yang Terlatih:
                <br>
                <code style="background-color: #F1F5F9; border: 2px solid #000000; padding: 6px 12px; font-weight: 900; display: block; margin: 10px 0; font-size: 0.8rem; text-align: center; color: #000000; box-shadow: 2px 2px 0px #000000;">
                    Keuntungan = {model.intercept_:.2f} + ({coef_iklan:.2f} × Iklan) + ({coef_diskon:.2f} × Diskon)
                </code>
            </p>
            <hr style="margin: 15px 0; border: 0; border-top: 2px solid #000000;">
            <h5 style="color: #000000; font-weight: 900; text-transform: uppercase; margin-bottom: 8px;">Pengaruh Variabel</h5>
            <ul style="padding-left: 20px; font-size: 0.85rem; color: #000000; line-height: 1.6; font-weight: 700;">
                <li><strong>Biaya Iklan:</strong> Setiap penambahan budget iklan sebesar <strong>Rp 1 Juta</strong> diproyeksikan akan meningkatkan keuntungan sebesar <strong>Rp {coef_iklan:.2f} Juta</strong>.</li>
                <li><strong>Diskon:</strong> Setiap penambahan persentase diskon sebesar <strong>1%</strong> diproyeksikan akan {'meningkatkan' if coef_diskon >= 0 else 'menurunkan'} keuntungan sebesar <strong>Rp {abs(coef_diskon):.2f} Juta</strong>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Informasi koefisien model tidak tersedia.")

    # Rekomendasi Pintar (Smart Recommendation) berbasis kondisi skenario
    st.markdown("### Rekomendasi Keputusan")
    
    if delta > 0:
        rec_box = f"""
        <div style="background-color: #FFFFFF; border: 4px solid #000000; border-radius: 0px; padding: 20px; font-size: 0.85rem; color: #000000; line-height: 1.5; box-shadow: 6px 6px 0px 0px #000000;">
            <div style="background-color: #4ADE80; display: inline-block; padding: 2px 8px; border: 2px solid #000000; font-weight: 900; text-transform: uppercase; margin-bottom: 10px; font-size: 0.75rem; box-shadow: 1px 1px 0px #000000;">Keputusan Menguntungkan</div>
            <br>
            <span style="font-weight: 700;">Kombinasi parameter saat ini menghasilkan estimasi keuntungan sebesar <strong>Rp {prediksi:.2f} Juta</strong>, yang mana <strong>lebih tinggi</strong> sebesar Rp {abs(delta):.2f} Juta dibanding kondisi awal. Rekomendasi untuk menerapkan kebijakan iklan Rp {iklan} Juta.</span>
        </div>
        """
    elif delta < 0:
        rec_box = f"""
        <div style="background-color: #FFFFFF; border: 4px solid #000000; border-radius: 0px; padding: 20px; font-size: 0.85rem; color: #000000; line-height: 1.5; box-shadow: 6px 6px 0px 0px #000000;">
            <div style="background-color: #F87171; display: inline-block; padding: 2px 8px; border: 2px solid #000000; font-weight: 900; text-transform: uppercase; margin-bottom: 10px; font-size: 0.75rem; box-shadow: 1px 1px 0px #000000;">Peringatan Kebijakan</div>
            <br>
            <span style="font-weight: 700;">Skenario baru ini menghasilkan penurunan profit sebesar <strong>Rp {abs(delta):.2f} Juta</strong> dibanding baseline. Hal ini kemungkinan disebabkan oleh diskon yang terlalu tinggi atau biaya iklan yang kurang optimal. Disarankan untuk menekan persentase diskon atau menaikkan budget iklan.</span>
        </div>
        """
    else:
        rec_box = """
        <div style="background-color: #FFFFFF; border: 4px solid #000000; border-radius: 0px; padding: 20px; font-size: 0.85rem; color: #000000; line-height: 1.5; box-shadow: 6px 6px 0px 0px #000000;">
            <div style="background-color: #E2E8F0; display: inline-block; padding: 2px 8px; border: 2px solid #000000; font-weight: 900; text-transform: uppercase; margin-bottom: 10px; font-size: 0.75rem; box-shadow: 1px 1px 0px #000000;">Kondisi Baseline</div>
            <br>
            <span style="font-weight: 700;">Kebijakan saat ini berjalan pada parameter baseline. Silakan sesuaikan slider di panel samping untuk melakukan eksperimen simulasi.</span>
        </div>
        """
    st.markdown(rec_box, unsafe_allow_html=True)