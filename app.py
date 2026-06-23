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

# 2. Injeksi CSS Custom untuk Tampilan Neobrutalism Modern & Soft
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
        font-size: 2.3rem;
        color: #1E293B !important; /* Paksa warna gelap agar terbaca pada background biru muda */
        text-align: center;
        text-transform: uppercase;
        letter-spacing: -1px;
        background-color: #EFF6FF; /* Soft Pastel Blue */
        border: 3px solid #1E293B; /* Soft Slate Border */
        padding: 15px 30px;
        box-shadow: 5px 5px 0px 0px #1E293B; /* Soft Slate Shadow */
        display: block;
        margin: 10px auto 10px auto;
        box-sizing: border-box;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF !important; /* Paksa warna putih agar terlihat di dark mode */
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 35px;
    }
    
    /* Card KPI */
    .kpi-card {
        background-color: #FFFFFF;
        border: 3px solid #1E293B;
        border-radius: 0px;
        padding: 24px;
        box-shadow: 5px 5px 0px 0px #1E293B;
        margin-bottom: 25px;
        color: #1E293B;
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    
    .kpi-card:hover {
        transform: translate(-2px, -2px);
        box-shadow: 7px 7px 0px 0px #1E293B;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 800;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #1E293B;
        padding-bottom: 5px;
        margin-bottom: 12px;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 900;
        color: #1E293B;
        margin: 10px 0;
    }
    
    /* Neobrutalist Badges */
    .delta-badge {
        display: inline-flex;
        align-items: center;
        font-size: 0.85rem;
        font-weight: 800;
        padding: 6px 14px;
        border: 2px solid #1E293B;
        border-radius: 0px;
        margin-top: 8px;
        text-transform: uppercase;
        box-shadow: 2px 2px 0px #1E293B;
    }
    
    .delta-positive {
        background-color: #D1FAE5; /* Soft Mint Green */
        color: #065F46; /* Dark Emerald Text */
    }
    
    .delta-negative {
        background-color: #FEE2E2; /* Soft Pastel Red */
        color: #991B1B; /* Dark Red Text */
    }
    
    /* Highlight model metadata info */
    .meta-info-box {
        font-size: 0.8rem;
        font-weight: 700;
        padding: 8px 12px;
        background-color: #F8FAFC;
        border: 2px solid #1E293B;
        border-radius: 0px;
        margin-top: 15px;
        color: #475569;
        text-transform: uppercase;
        box-shadow: 2px 2px 0px #1E293B;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar - Profile & Input
# Header Profil Neobrutalist Modern & Soft
st.sidebar.markdown(f"""
<div style="background-color: #FFFFFF; border: 3px solid #1E293B; border-radius: 0px; padding: 20px; text-align: center; margin-bottom: 25px; box-shadow: 4px 4px 0px 0px #1E293B; color: #1E293B;">
    <div style="width: 55px; height: 55px; background-color: #DBEAFE; color: #1E40AF; border: 3px solid #1E293B; border-radius: 0px; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px auto; font-size: 1.3rem; font-weight: 900; box-shadow: 2px 2px 0px #1E293B;">
        BAS
    </div>
    <h4 style="margin: 0; font-weight: 900; font-size: 1.05rem; color: #1E293B; text-transform: uppercase;">Bagas Aji Saputra</h4>
    <p style="margin: 3px 0 0 0; font-size: 0.8rem; color: #475569; font-weight: 700; opacity: 0.9;">NPM 2313020028</p>
    <div style="margin-top: 12px; display: inline-block; font-size: 0.75rem; color: #9D174D; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; background-color: #FCE7F3; border: 2px solid #1E293B; padding: 3px 10px; box-shadow: 2px 2px 0px #1E293B;">
        PNS Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Input Kebijakan")
st.sidebar.markdown("<p style='font-size:0.85rem; opacity:0.9; margin-top:-10px; font-weight:600; color:#E2E8F0;'>Sesuaikan parameter untuk memicu prediksi keuntungan baru.</p>", unsafe_allow_html=True)

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

# Tombol reset
st.sidebar.button("Reset ke Baseline", on_click=reset_to_baseline, use_container_width=True)

# Dropdown Simulasi Anomali Eksternal
st.sidebar.markdown("### Simulasi Anomali Eksternal")
anomaly_option = st.sidebar.selectbox(
    "Pilih Kejadian Pasar",
    options=[
        "Normal (Tanpa Anomali)",
        "Invasi Kucing Oren",
        "Kesurupan Tren TikTok",
        "Mesin Kopi Kantor Rusak",
        "Meme Coin Stonks Meroket",
        "Merkurius Sedang Retrograde"
    ],
    index=0
)

# Multiplier dan deskripsi anomali
anomaly_data = {
    "Normal (Tanpa Anomali)": {"multiplier": 1.0, "desc": "Kondisi pasar stabil. Kopi aman mengalir, kucing tertib.", "color": "#1E293B"},
    "Invasi Kucing Oren": {"multiplier": 0.85, "desc": "Kucing-kucing oren tidur di keyboard & tidak sengaja menghapus database marketing. Traffic naik sedikit karena kelucuan mereka.", "color": "#F59E0B"},
    "Kesurupan Tren TikTok": {"multiplier": 1.45, "desc": "Seluruh tim sales joget viral di FYP. Produktivitas formal turun 90%, tapi profit melonjak karena viral gratis.", "color": "#EC4899"},
    "Mesin Kopi Kantor Rusak": {"multiplier": 0.4, "desc": "Mesin kopi rusak total. Seluruh tim mengantuk berat. Developer salah push code, iklan salah target ke negara antah berantah.", "color": "#78350F"},
    "Meme Coin Stonks Meroket": {"multiplier": 1.95, "desc": "Perusahaan iseng terima pembayaran koin meme bergambar anjing. Koin tersebut naik 1000% setelah di-tweet tokoh terkenal.", "color": "#10B981"},
    "Merkurius Sedang Retrograde": {"multiplier": 0.9, "desc": "Server error misterius, printer nge-print dokumen aneh sendiri. Tim pasrah, tapi untungnya pelanggan beli karena kasihan.", "color": "#6366F1"}
}

active_anomaly = anomaly_data[anomaly_option]
multiplier = active_anomaly["multiplier"]

st.sidebar.markdown(f"""
<div style="background-color: #F8FAFC; border: 2px solid #1E293B; padding: 12px; font-size: 0.8rem; font-weight: 700; color: #1E293B; margin-top: 10px; box-shadow: 3px 3px 0px #1E293B;">
    <span style="color: {active_anomaly['color']}; text-transform: uppercase;">{anomaly_option}</span><br>
    <p style="margin: 5px 0 0 0; font-weight: 500; font-size: 0.75rem; color: #475569; line-height: 1.4;">{active_anomaly['desc']}</p>
</div>
""", unsafe_allow_html=True)

# Logika Deteksi Anomali Kebijakan (Internal)
policy_anomaly = None
if iklan > 40 and diskon > 15:
    policy_anomaly = {
        "title": "Sultan Gabut / Bakar Uang",
        "desc": "Anda mengalokasikan anggaran iklan sangat besar (> Rp 40 Jt) sekaligus memberikan diskon tinggi (> 15%). Investor menangis melihat bakar uang ini demi validasi mantan.",
        "bg": "#FEE2E2",
        "border": "#EF4444",
        "text": "#991B1B"
    }
elif iklan < 3 and diskon < 3:
    policy_anomaly = {
        "title": "Strategi Goib (Ghosting Market)",
        "desc": "Anggaran iklan (< Rp 3 Jt) dan diskon (< 3%) sangat rendah. Berjualan dengan asas keikhlasan tingkat tinggi. Produk Anda tak terlihat bagaikan makhluk halus.",
        "bg": "#FEF3C7",
        "border": "#D97706",
        "text": "#92400E"
    }
elif iklan < 8 and diskon > 15:
    policy_anomaly = {
        "title": "Bagi-Bagi Bansos / Sedekah Ekstrim",
        "desc": "Anda memberikan diskon besar (> 15%) tanpa dukungan budget iklan yang cukup (< Rp 8 Jt). Anda tidak sedang berbisnis, melainkan membagikan sembako gratis. Pembeli senang, dompet perusahaan meriang.",
        "bg": "#F5F3FF",
        "border": "#8B5CF6",
        "text": "#5B21B6"
    }
elif iklan > 45 and diskon < 3:
    policy_anomaly = {
        "title": "Spammer Garis Keras",
        "desc": "Anggaran iklan sangat besar (> Rp 45 Jt) tetapi diskon sangat minim (< 3%). Feed media sosial pelanggan penuh dengan iklan Anda sampai mereka memblokir Anda.",
        "bg": "#ECFDF5",
        "border": "#10B981",
        "text": "#065F46"
    }

# 4. Prediksi (What-If Analysis)
input_data = np.array([[iklan, diskon]])
prediksi_base = model.predict(input_data)[0]
prediksi = prediksi_base * multiplier

# Hitung selisih dibandingkan baseline yang dihitung model
delta = prediksi - baseline_pred
delta_pct = (delta / baseline_pred) * 100 if baseline_pred != 0 else 0

# Definisikan warna & icon berdasarkan arah delta (Menggunakan palette modern soft)
if delta >= 0:
    delta_class = "delta-positive"
    delta_text = "+"
    delta_pct_str = f"+{delta_pct:.2f}%"
    chart_highlight_color = "#10B981" # Modern Emerald Green
else:
    delta_class = "delta-negative"
    delta_text = ""
    delta_pct_str = f"{delta_pct:.2f}%"
    chart_highlight_color = "#EF4444" # Modern Rose Red

# 5. Tampilan Main Panel
st.markdown('<h1 class="main-title">DASHBOARD SIMULASI KEUNTUNGAN</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Platform Simulasi Prediktif untuk Pengambilan Keputusan Bisnis Secara Pintar</p>', unsafe_allow_html=True)

# Tampilkan Deteksi Anomali Kebijakan jika ada
if policy_anomaly:
    st.markdown(f"""
    <div style="background-color: {policy_anomaly['bg']}; border: 3px solid {policy_anomaly['border']}; padding: 18px; margin-bottom: 25px; box-shadow: 4px 4px 0px 0px #1E293B; color: #1E293B;">
        <span style="background-color: {policy_anomaly['border']}; color: #FFFFFF; font-size: 0.75rem; font-weight: 900; padding: 4px 10px; border: 2px solid #1E293B; box-shadow: 1px 1px 0px #1E293B; text-transform: uppercase;">
            ⚠️ Terdeteksi Anomali Kebijakan: {policy_anomaly['title']}
        </span>
        <p style="margin: 12px 0 0 0; font-size: 0.85rem; font-weight: 700; line-height: 1.5; color: {policy_anomaly['text']};">
            {policy_anomaly['desc']}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tata Letak Grid: Tiga Card KPI Kolom di Bagian Atas
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    if multiplier != 1.0:
        kpi_title = f"Skenario {anomaly_option}"
        kpi_value_html = f"""
        <div class="kpi-value" style="color: #E11D48; font-size: 1.75rem; margin: 10px 0 5px 0;">Rp {prediksi:.2f} Juta</div>
        <div style="font-size: 0.75rem; font-weight: 800; color: #475569; margin-top: -5px; text-transform: uppercase; border-top: 1px solid #E2E8F0; padding-top: 5px;">
            Base Model: Rp {prediksi_base:.2f} Jt
        </div>
        """
    else:
        kpi_title = "Skenario Baru (Prediksi)"
        kpi_value_html = f'<div class="kpi-value" style="color: #2563EB; margin: 10px 0;">Rp {prediksi:.2f} Juta</div>'

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{kpi_title}</div>
        {kpi_value_html}
        <div class="meta-info-box">
            Iklan: {iklan} Jt • Diskon: {diskon}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Baseline (Kondisi Awal)</div>
        <div class="kpi-value" style="color: #1E293B; opacity: 0.95; margin: 10px 0;">Rp {baseline_pred:.2f} Juta</div>
        <div class="meta-info-box" style="background-color: #F8FAFC;">
            Iklan: 10 Jt • Diskon: 10%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Estimasi Dampak Profit</div>
        <div class="kpi-value" style="font-size: 2rem; font-weight: 900; color: #1E293B; margin: 10px 0;">
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
        # Membuat visualisasi trend 2D berkualitas tinggi neobrutalist modern & soft
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
        fig.patch.set_facecolor('none')  # Background transparan agar menyatu dengan Streamlit
        
        # --- Subplot 1: Keuntungan vs Biaya Iklan ---
        ax1.set_facecolor('#FFFFFF')
        iklan_range = np.linspace(0, 50, 100)
        pred_line_iklan_base = model.predict(np.column_stack((iklan_range, np.full_like(iklan_range, diskon))))
        
        # Garis Regresi Model (Base)
        ax1.plot(iklan_range, pred_line_iklan_base, color='#94A3B8', linestyle='--', linewidth=2, label='Regresi Normal')
        
        if multiplier != 1.0:
            # Garis Regresi Terkoreksi Anomali
            pred_line_iklan_anomaly = pred_line_iklan_base * multiplier
            ax1.plot(iklan_range, pred_line_iklan_anomaly, color=active_anomaly["color"], linewidth=3, label=f'Regresi {anomaly_option}')
            # Titik Prediksi Normal (tanpa anomali)
            ax1.scatter(iklan, prediksi_base, color='#E2E8F0', edgecolors='#1E293B', linewidths=1.5, s=120, marker='o', zorder=4, label='Prediksi Normal')
        else:
            # Garis utama biru solid jika normal
            ax1.plot(iklan_range, pred_line_iklan_base, color='#3B82F6', linewidth=3, label='Regresi Aktif')
        
        # Scatter data training asli (Neobrutalist soft dengan outline slate)
        ax1.scatter(X_train[:, 0], y_train, color='#94A3B8', edgecolors='#1E293B', linewidths=2, s=60, alpha=1.0, zorder=3, label='Data Historis')
        
        # Sorot titik pilihan user saat ini (Bintang besar dengan outline slate)
        ax1.scatter(iklan, prediksi, color=chart_highlight_color, edgecolors='#1E293B', linewidths=2, s=250, marker='*', zorder=5, label='Skenario Anda')
        
        # Mengubah warna teks pada grafik menjadi putih agar terlihat di dark mode
        ax1.set_title("Keuntungan vs Biaya Iklan\n(Diskon Tetap)", fontsize=9, fontweight='black', color='#FFFFFF')
        ax1.set_xlabel("Biaya Iklan (Juta)", fontsize=8, fontweight='bold', color='#FFFFFF')
        ax1.set_ylabel("Keuntungan (Juta)", fontsize=8, fontweight='bold', color='#FFFFFF')
        ax1.grid(True, linestyle='-', alpha=0.15, color='#FFFFFF')
        ax1.tick_params(labelsize=8, colors='#FFFFFF')
        
        # Spines (bingkai) putih tebal agar terlihat di background gelap
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_color('#FFFFFF')
            spine.set_linewidth(2.2)
        
        # --- Subplot 2: Keuntungan vs Diskon ---
        ax2.set_facecolor('#FFFFFF')
        diskon_range = np.linspace(0, 20, 100)
        pred_line_diskon_base = model.predict(np.column_stack((np.full_like(diskon_range, iklan), diskon_range)))
        
        # Garis Regresi Model (Base)
        ax2.plot(diskon_range, pred_line_diskon_base, color='#94A3B8', linestyle='--', linewidth=2)
        
        if multiplier != 1.0:
            # Garis Regresi Terkoreksi Anomali
            pred_line_diskon_anomaly = pred_line_diskon_base * multiplier
            ax2.plot(diskon_range, pred_line_diskon_anomaly, color=active_anomaly["color"], linewidth=3)
            # Titik Prediksi Normal (tanpa anomali)
            ax2.scatter(diskon, prediksi_base, color='#E2E8F0', edgecolors='#1E293B', linewidths=1.5, s=120, marker='o', zorder=4)
        else:
            ax2.plot(diskon_range, pred_line_diskon_base, color='#3B82F6', linewidth=3)
        
        # Scatter data training asli
        ax2.scatter(X_train[:, 1], y_train, color='#94A3B8', edgecolors='#1E293B', linewidths=2, s=60, alpha=1.0, zorder=3)
        
        # Sorot titik pilihan user saat ini
        ax2.scatter(diskon, prediksi, color=chart_highlight_color, edgecolors='#1E293B', linewidths=2, s=250, marker='*', zorder=5)
        
        ax2.set_title("Keuntungan vs Diskon\n(Iklan Tetap)", fontsize=9, fontweight='black', color='#FFFFFF')
        ax2.set_xlabel("Diskon (%)", fontsize=8, fontweight='bold', color='#FFFFFF')
        ax2.set_ylabel("Keuntungan (Juta)", fontsize=8, fontweight='bold', color='#FFFFFF')
        ax2.grid(True, linestyle='-', alpha=0.15, color='#FFFFFF')
        ax2.tick_params(labelsize=8, colors='#FFFFFF')
        
        # Spines putih tebal
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_color('#FFFFFF')
            spine.set_linewidth(2.2)
        
        # Handler legend satu untuk keseluruhan chart
        fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', ncol=4, frameon=True, edgecolor='#FFFFFF', facecolor='#1E293B', fontsize=8, labelcolor='white')
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        st.pyplot(fig)
        
    with tab2:
        # Grafik Komparasi Baseline vs Skenario Baru
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3.5), dpi=150)
        fig_bar.patch.set_facecolor('none')
        ax_bar.set_facecolor('#FFFFFF')
        
        if multiplier != 1.0:
            labels_bar = ['Baseline', 'Skenario Normal', f'Skenario + {anomaly_option}']
            values_bar = [baseline_pred, prediksi_base, prediksi]
            colors_bar = ['#E2E8F0', '#DBEAFE', active_anomaly["color"]]
        else:
            labels_bar = ['Baseline', 'Skenario Baru']
            values_bar = [baseline_pred, prediksi]
            colors_bar = ['#E2E8F0', '#DBEAFE']
            
        # Bar horizontal neobrutalist soft dengan border slate tebal
        bars = ax_bar.barh(labels_bar, values_bar, height=0.5, color=colors_bar, edgecolor='#1E293B', linewidth=2.2)
        
        # Custom styling bar chart horizontal
        for spine in ax_bar.spines.values():
            spine.set_visible(True)
            spine.set_color('#FFFFFF')
            spine.set_linewidth(2.2)
            
        ax_bar.set_xlabel("Keuntungan (Juta Rupiah)", fontsize=8, fontweight='bold', color='#FFFFFF')
        ax_bar.tick_params(labelsize=8, colors='#FFFFFF')
        ax_bar.grid(axis='x', linestyle='-', alpha=0.15, color='#FFFFFF')
        
        # Menambahkan label teks di dalam/ujung bar dengan warna putih agar kontras di dark background
        for bar in bars:
            width = bar.get_width()
            ax_bar.text(width + 2, bar.get_y() + bar.get_height()/2, f'Rp {width:.2f} Jt', 
                        ha='left', va='center', fontweight='black', fontsize=8, color='#FFFFFF')
            
        plt.tight_layout()
        st.pyplot(fig_bar)

with col_insight:
    st.markdown("### Wawasan & Analisis Model")
    
    # Menampilkan interpretasi model regresi linier secara dinamis
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        coef_iklan = model.coef_[0]
        coef_diskon = model.coef_[1]
        
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 3px solid #1E293B; border-radius: 0px; padding: 24px; box-shadow: 5px 5px 0px 0px #1E293B; color: #1E293B;">
            <h5 style="margin-top: 0; color: #1E293B; font-weight: 900; text-transform: uppercase; border-bottom: 2px solid #1E293B; padding-bottom: 5px; margin-bottom: 12px;">Karakteristik Model</h5>
            <p style="font-size: 0.85rem; line-height: 1.5; color: #1E293B; font-weight: 700;">
                Persamaan Regresi Linier yang Terlatih:
                <br>
                <code style="background-color: #F8FAFC; border: 2px solid #1E293B; padding: 6px 12px; font-weight: 900; display: block; margin: 10px 0; font-size: 0.8rem; text-align: center; color: #1E293B; box-shadow: 2px 2px 0px #1E293B;">
                    Keuntungan = {model.intercept_:.2f} + ({coef_iklan:.2f} × Iklan) + ({coef_diskon:.2f} × Diskon)
                </code>
            </p>
            <hr style="margin: 15px 0; border: 0; border-top: 2px solid #1E293B;">
            <h5 style="color: #1E293B; font-weight: 900; text-transform: uppercase; margin-bottom: 8px;">Pengaruh Variabel</h5>
            <ul style="padding-left: 20px; font-size: 0.85rem; color: #1E293B; line-height: 1.6; font-weight: 700;">
                <li><strong>Biaya Iklan:</strong> Setiap penambahan budget iklan sebesar <strong>Rp 1 Juta</strong> diproyeksikan akan meningkatkan keuntungan sebesar <strong>Rp {coef_iklan:.2f} Juta</strong>.</li>
                <li><strong>Diskon:</strong> Setiap penambahan persentase diskon sebesar <strong>1%</strong> diproyeksikan akan {'meningkatkan' if coef_diskon >= 0 else 'menurunkan'} keuntungan sebesar <strong>Rp {abs(coef_diskon):.2f} Juta</strong>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Informasi koefisien model tidak tersedia.")

    # Rekomendasi Pintar (Smart Recommendation) berbasis kondisi skenario
    st.markdown("### Rekomendasi Keputusan")
    
    if multiplier != 1.0:
        if multiplier < 1.0:
            rec_box = f"""
            <div style="background-color: #FFFFFF; border: 3px solid #1E293B; border-radius: 0px; padding: 20px; font-size: 0.85rem; color: #1E293B; line-height: 1.5; box-shadow: 5px 5px 0px 0px #1E293B;">
                <div style="background-color: #FEE2E2; color: #991B1B; display: inline-block; padding: 2px 8px; border: 2px solid #1E293B; font-weight: 900; text-transform: uppercase; margin-bottom: 10px; font-size: 0.75rem; box-shadow: 1px 1px 0px #1E293B;">⚠️ Peringatan Kekacauan Eksternal</div>
                <br>
                <span style="font-weight: 700;">Dampak dari <strong>{anomaly_option}</strong> memangkas profit potensial sebesar {100*(1-multiplier):.0f}% menjadi <strong>Rp {prediksi:.2f} Juta</strong>. Hubungi pawang kucing atau perbaiki mesin kopi segera demi menyelamatkan keuangan perusahaan!</span>
            </div>
            """
        else:
            rec_box = f"""
            <div style="background-color: #FFFFFF; border: 3px solid #1E293B; border-radius: 0px; padding: 20px; font-size: 0.85rem; color: #1E293B; line-height: 1.5; box-shadow: 5px 5px 0px 0px #1E293B;">
                <div style="background-color: #D1FAE5; color: #065F46; display: inline-block; padding: 2px 8px; border: 2px solid #1E293B; font-weight: 900; text-transform: uppercase; margin-bottom: 10px; font-size: 0.75rem; box-shadow: 1px 1px 0px #1E293B;">🚀 Keberuntungan Haqiqi</div>
                <br>
                <span style="font-weight: 700;">Hore! Efek dari <strong>{anomaly_option}</strong> melipatgandakan profit Anda sebesar {100*(multiplier-1):.0f}% di atas ekspektasi model menjadi <strong>Rp {prediksi:.2f} Juta</strong>. Segera cairkan koin meme Anda atau teruslah berjoget di TikTok sebelum algoritma berubah!</span>
            </div>
            """
    else:
        if delta > 0:
            rec_box = f"""
            <div style="background-color: #FFFFFF; border: 3px solid #1E293B; border-radius: 0px; padding: 20px; font-size: 0.85rem; color: #1E293B; line-height: 1.5; box-shadow: 5px 5px 0px 0px #1E293B;">
                <div style="background-color: #D1FAE5; color: #065F46; display: inline-block; padding: 2px 8px; border: 2px solid #1E293B; font-weight: 900; text-transform: uppercase; margin-bottom: 10px; font-size: 0.75rem; box-shadow: 1px 1px 0px #1E293B;">Keputusan Menguntungkan</div>
                <br>
                <span style="font-weight: 700;">Kombinasi parameter saat ini menghasilkan estimasi keuntungan sebesar <strong>Rp {prediksi:.2f} Juta</strong>, yang mana <strong>lebih tinggi</strong> sebesar Rp {abs(delta):.2f} Juta dibanding kondisi awal. Rekomendasi untuk menerapkan kebijakan iklan Rp {iklan} Juta.</span>
            </div>
            """
        elif delta < 0:
            rec_box = f"""
            <div style="background-color: #FFFFFF; border: 3px solid #1E293B; border-radius: 0px; padding: 20px; font-size: 0.85rem; color: #1E293B; line-height: 1.5; box-shadow: 5px 5px 0px 0px #1E293B;">
                <div style="background-color: #FEE2E2; color: #991B1B; display: inline-block; padding: 2px 8px; border: 2px solid #1E293B; font-weight: 900; text-transform: uppercase; margin-bottom: 10px; font-size: 0.75rem; box-shadow: 1px 1px 0px #1E293B;">Peringatan Kebijakan</div>
                <br>
                <span style="font-weight: 700;">Skenario baru ini menghasilkan penurunan profit sebesar <strong>Rp {abs(delta):.2f} Juta</strong> dibanding baseline. Hal ini kemungkinan disebabkan oleh diskon yang terlalu tinggi atau biaya iklan yang kurang optimal. Disarankan untuk menekan persentase diskon atau menaikkan budget iklan.</span>
            </div>
            """
        else:
            rec_box = """
            <div style="background-color: #FFFFFF; border: 3px solid #1E293B; border-radius: 0px; padding: 20px; font-size: 0.85rem; color: #1E293B; line-height: 1.5; box-shadow: 5px 5px 0px 0px #1E293B;">
                <div style="background-color: #F1F5F9; color: #334155; display: inline-block; padding: 2px 8px; border: 2px solid #1E293B; font-weight: 900; text-transform: uppercase; margin-bottom: 10px; font-size: 0.75rem; box-shadow: 1px 1px 0px #1E293B;">Kondisi Baseline</div>
                <br>
                <span style="font-weight: 700;">Kebijakan saat ini berjalan pada parameter baseline. Silakan sesuaikan slider di panel samping untuk melakukan eksperimen simulasi.</span>
            </div>
            """
    st.markdown(rec_box, unsafe_allow_html=True)