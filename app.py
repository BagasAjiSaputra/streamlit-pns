import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Konfigurasi Halaman (Harus dipanggil pertama kali sebelum elemen Streamlit lain)
st.set_page_config(
    page_title="Dashboard Prediksi Keuntungan",
    page_icon="📈",
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

# 2. Injeksi CSS Custom untuk Tampilan Clean & Modern
st.markdown("""
<style>
    /* Styling font global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Judul Utama */
    .main-title {
        background: linear-gradient(135deg, #2e7bcf 0%, #1a365d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 25px;
    }
    
    /* Card KPI */
    .kpi-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 10px;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 8px 0;
        color: var(--text-color);
    }
    
    /* Delta indicator badges */
    .delta-badge {
        display: inline-flex;
        align-items: center;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        margin-top: 4px;
    }
    
    .delta-positive {
        color: #10B981;
        background-color: rgba(16, 185, 129, 0.1);
    }
    
    .delta-negative {
        color: #EF4444;
        background-color: rgba(239, 68, 68, 0.1);
    }
    
    /* Highlight model metadata info */
    .meta-info-box {
        font-size: 0.85rem;
        padding: 12px 16px;
        border-radius: 8px;
        background-color: rgba(46, 123, 207, 0.05);
        border-left: 4px solid #2e7bcf;
        margin-top: 15px;
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar - Profile & Input
# Header Profil
st.sidebar.markdown(f"""
<div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.15); border-radius: 16px; padding: 20px; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.02);">
    <div style="width: 50px; height: 50px; background-color: #2e7bcf; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px auto; font-size: 1.2rem; font-weight: bold; box-shadow: 0 4px 10px rgba(46, 123, 207, 0.3);">
        FA
    </div>
    <h4 style="margin: 0; font-weight: 700; font-size: 1.05rem; color: var(--text-color);">Fatwabith Akbar</h4>
    <p style="margin: 3px 0 0 0; font-size: 0.8rem; color: var(--text-color); opacity: 0.6;">NPM: 2313020014</p>
    <div style="margin-top: 10px; display: inline-block; font-size: 0.7rem; color: #2e7bcf; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; background-color: rgba(46, 123, 207, 0.1); padding: 2px 8px; border-radius: 12px;">
        PNS Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🎛️ Input Kebijakan")
st.sidebar.markdown("<p style='font-size:0.85rem; opacity:0.8; margin-top:-10px;'>Sesuaikan parameter untuk memicu prediksi keuntungan baru.</p>", unsafe_allow_html=True)

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

# Tombol reset dengan container width
st.sidebar.button("🔄 Reset ke Baseline", on_click=reset_to_baseline, use_container_width=True)

# 4. Prediksi (What-If Analysis)
input_data = np.array([[iklan, diskon]])
prediksi = model.predict(input_data)[0]

# Hitung selisih dibandingkan baseline yang dihitung model
delta = prediksi - baseline_pred
delta_pct = (delta / baseline_pred) * 100 if baseline_pred != 0 else 0

# Definisikan warna & icon berdasarkan arah delta
if delta >= 0:
    delta_class = "delta-positive"
    delta_icon = "📈 +"
    delta_pct_str = f"+{delta_pct:.2f}%"
    chart_highlight_color = "#10B981" # Emerald
else:
    delta_class = "delta-negative"
    delta_icon = "📉 "
    delta_pct_str = f"{delta_pct:.2f}%"
    chart_highlight_color = "#EF4444" # Red

# 5. Tampilan Main Panel
st.markdown('<h1 class="main-title">📈 Dashboard Simulasi Keuntungan</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Platform Simulasi Prediktif untuk Pengambilan Keputusan Bisnis Secara Pintar</p>', unsafe_allow_html=True)

# Tata Letak Grid: Tiga Card KPI Kolom di Bagian Atas
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Skenario Baru (Prediksi)</div>
        <div class="kpi-value" style="color: #2e7bcf;">Rp {prediksi:.2f} Juta</div>
        <div class="meta-info-box" style="margin-top: 5px; padding: 6px 12px; font-size: 0.8rem;">
            Iklan: {iklan} Jt • Diskon: {diskon}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Baseline (Kondisi Awal)</div>
        <div class="kpi-value" style="opacity: 0.85;">Rp {baseline_pred:.2f} Juta</div>
        <div class="meta-info-box" style="margin-top: 5px; padding: 6px 12px; font-size: 0.8rem; background-color: rgba(128,128,128,0.05); border-left-color: #718096;">
            Iklan: 10 Jt • Diskon: 10%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Estimasi Dampak Profit</div>
        <div class="kpi-value {delta_class}" style="font-size: 1.7rem; background: none; padding: 0; margin: 8px 0;">
            {delta_icon}{abs(delta):.2f} Juta
        </div>
        <span class="delta-badge {delta_class}">{delta_pct_str} dibanding baseline</span>
    </div>
    """, unsafe_allow_html=True)

# Pembatas Konten
st.markdown("<br>", unsafe_allow_html=True)

# Grid Visualisasi & Detail Insight
col_viz, col_insight = st.columns([1.6, 1])

with col_viz:
    st.markdown("### 📊 Visualisasi & Analisis Tren")
    
    # Penggunaan Tab agar visualisasi bersih dan terstruktur
    tab1, tab2 = st.tabs(["📈 Analisis Sensitivitas & Historis", "📊 Perbandingan Skenario"])
    
    with tab1:
        # Membuat visualisasi trend 2D berkualitas tinggi
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
        fig.patch.set_facecolor('none')  # Background transparan agar menyatu dengan Streamlit
        
        # --- Subplot 1: Keuntungan vs Biaya Iklan ---
        ax1.set_facecolor('none')
        # Line prediksi model (diskon tetap sesuai input slider)
        iklan_range = np.linspace(0, 50, 100)
        pred_line_iklan = model.predict(np.column_stack((iklan_range, np.full_like(iklan_range, diskon))))
        ax1.plot(iklan_range, pred_line_iklan, color='#2e7bcf', linewidth=2, label='Garis Regresi')
        
        # Scatter data training asli
        ax1.scatter(X_train[:, 0], y_train, color='#718096', s=45, alpha=0.8, edgecolors='w', linewidth=0.8, label='Data Historis')
        
        # Sorot titik pilihan user saat ini
        ax1.scatter(iklan, prediksi, color=chart_highlight_color, s=130, marker='*', zorder=5, label='Skenario Anda')
        
        ax1.set_title("Keuntungan vs Biaya Iklan\n(Diskon Tetap)", fontsize=9, fontweight='bold', color='#4A5568')
        ax1.set_xlabel("Biaya Iklan (Juta)", fontsize=8, color='#4A5568')
        ax1.set_ylabel("Keuntungan (Juta)", fontsize=8, color='#4A5568')
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E0')
        ax1.tick_params(labelsize=8, colors='#4A5568')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#CBD5E0')
        ax1.spines['bottom'].set_color('#CBD5E0')
        
        # --- Subplot 2: Keuntungan vs Diskon ---
        ax2.set_facecolor('none')
        # Line prediksi model (iklan tetap sesuai input slider)
        diskon_range = np.linspace(0, 20, 100)
        pred_line_diskon = model.predict(np.column_stack((np.full_like(diskon_range, iklan), diskon_range)))
        ax2.plot(diskon_range, pred_line_diskon, color='#2e7bcf', linewidth=2)
        
        # Scatter data training asli
        ax2.scatter(X_train[:, 1], y_train, color='#718096', s=45, alpha=0.8, edgecolors='w', linewidth=0.8)
        
        # Sorot titik pilihan user saat ini
        ax2.scatter(diskon, prediksi, color=chart_highlight_color, s=130, marker='*', zorder=5)
        
        ax2.set_title("Keuntungan vs Diskon\n(Iklan Tetap)", fontsize=9, fontweight='bold', color='#4A5568')
        ax2.set_xlabel("Diskon (%)", fontsize=8, color='#4A5568')
        ax2.set_ylabel("Keuntungan (Juta)", fontsize=8, color='#4A5568')
        ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E0')
        ax2.tick_params(labelsize=8, colors='#4A5568')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#CBD5E0')
        ax2.spines['bottom'].set_color('#CBD5E0')
        
        # Handler legend satu untuk keseluruhan chart
        fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False, fontsize=8)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        st.pyplot(fig)
        
    with tab2:
        # Grafik Komparasi Baseline vs Skenario Baru yang disempurnakan
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3), dpi=150)
        fig_bar.patch.set_facecolor('none')
        ax_bar.set_facecolor('none')
        
        bars = ax_bar.barh(['Baseline', 'Skenario Baru'], [baseline_pred, prediksi], height=0.5, color=['#A0AEC0', '#2e7bcf'])
        
        # Custom styling bar chart horizontal
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['bottom'].set_visible(False)
        ax_bar.spines['left'].set_color('#CBD5E0')
        ax_bar.set_xlabel("Keuntungan (Juta Rupiah)", fontsize=8, color='#4A5568')
        ax_bar.tick_params(labelsize=8, colors='#4A5568')
        ax_bar.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E0')
        
        # Menambahkan label teks di dalam/ujung bar
        for bar in bars:
            width = bar.get_width()
            ax_bar.text(width + 2, bar.get_y() + bar.get_height()/2, f'Rp {width:.2f} Jt', 
                        ha='left', va='center', fontweight='bold', fontsize=8, color='#2D3748')
            
        plt.tight_layout()
        st.pyplot(fig_bar)

with col_insight:
    st.markdown("### 💡 Wawasan & Analisis Model")
    
    # Menampilkan interpretasi model regresi linier secara dinamis
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        coef_iklan = model.coef_[0]
        coef_diskon = model.coef_[1]
        
        st.markdown(f"""
        <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.1); border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
            <h5 style="margin-top: 0; color: var(--text-color); font-weight: 600;">📊 Karakteristik Model</h5>
            <p style="font-size: 0.85rem; line-height: 1.5; color: var(--text-color); opacity: 0.85;">
                Persamaan Regresi Linier yang Terlatih:
                <br>
                <code style="background-color: rgba(128,128,128,0.1); padding: 2px 6px; border-radius: 4px; font-weight: bold; display: block; margin: 8px 0; font-size: 0.8rem; text-align: center;">
                    Keuntungan = {model.intercept_:.2f} + ({coef_iklan:.2f} × Iklan) + ({coef_diskon:.2f} × Diskon)
                </code>
            </p>
            <hr style="margin: 15px 0; border: 0; border-top: 1px solid rgba(128,128,128,0.15);">
            <h5 style="color: var(--text-color); font-weight: 600;">📈 Pengaruh Variabel</h5>
            <ul style="padding-left: 20px; font-size: 0.85rem; color: var(--text-color); opacity: 0.85; line-height: 1.6;">
                <li><strong>Biaya Iklan:</strong> Setiap penambahan budget iklan sebesar <strong>Rp 1 Juta</strong> diproyeksikan akan meningkatkan keuntungan sebesar <strong>Rp {coef_iklan:.2f} Juta</strong>.</li>
                <li><strong>Diskon:</strong> Setiap penambahan persentase diskon sebesar <strong>1%</strong> diproyeksikan akan {'meningkatkan' if coef_diskon >= 0 else 'menurunkan'} keuntungan sebesar <strong>Rp {abs(coef_diskon):.2f} Juta</strong>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Informasi koefisien model tidak tersedia.")

    # Rekomendasi Pintar (Smart Recommendation) berbasis kondisi skenario
    st.markdown("### 🎯 Rekomendasi Keputusan")
    
    if delta > 0:
        rec_box = f"""
        <div style="background-color: rgba(16, 185, 129, 0.05); border-left: 4px solid #10B981; border-radius: 6px; padding: 15px; font-size: 0.85rem; color: var(--text-color); line-height: 1.5;">
            <strong>Keputusan Menguntungkan:</strong> Kombinasi parameter saat ini menghasilkan estimasi keuntungan sebesar <strong>Rp {prediksi:.2f} Juta</strong>, yang mana <strong>lebih tinggi</strong> sebesar Rp {abs(delta):.2f} Juta dibanding kondisi awal. Rekomendasi untuk menerapkan kebijakan iklan Rp {iklan} Juta.
        </div>
        """
    elif delta < 0:
        rec_box = f"""
        <div style="background-color: rgba(239, 68, 68, 0.05); border-left: 4px solid #EF4444; border-radius: 6px; padding: 15px; font-size: 0.85rem; color: var(--text-color); line-height: 1.5;">
            <strong>Peringatan Kebijakan:</strong> Skenario baru ini menghasilkan penurunan profit sebesar <strong>Rp {abs(delta):.2f} Juta</strong> dibanding baseline. Hal ini kemungkinan disebabkan oleh diskon yang terlalu tinggi atau biaya iklan yang kurang optimal. Disarankan untuk menekan persentase diskon atau menaikkan budget iklan.
        </div>
        """
    else:
        rec_box = """
        <div style="background-color: rgba(113, 128, 150, 0.05); border-left: 4px solid #718096; border-radius: 6px; padding: 15px; font-size: 0.85rem; color: var(--text-color); line-height: 1.5;">
            <strong>Kondisi Baseline:</strong> Kebijakan saat ini berjalan pada parameter baseline. Silakan sesuaikan slider di panel samping untuk melakukan eksperimen simulasi.
        </div>
        """
    st.markdown(rec_box, unsafe_allow_html=True)