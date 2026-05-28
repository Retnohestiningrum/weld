import streamlit as st
from PIL import Image
from ssd import SSD
import warnings

st.set_page_config(page_title="Deteksi Cacat Pengelasan - SSD ResNet50",
                   layout="centered", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #4682B4; }
    .stButton>button { background-color: #2b7a78; color: white; border-radius: 8px; padding: 0.5em 1em; font-weight: bold; }
    .stSlider>div>div>div { background-color: #3aafa9; }
    </style>
    """, unsafe_allow_html=True)


class Sidebar():
    def __init__(self):
        st.sidebar.image(
            'img/cacat.png',
            width="stretch"
        )
        st.sidebar.markdown(
            "### Atur Konfigurasi?")
        self.confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
        self.iou_threshold = st.sidebar.slider(
            "IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.01)


warnings.filterwarnings("ignore")
sb = Sidebar()


@st.cache_resource
def load_ssd_model(confidence, iou):
    return SSD(confidence=confidence, nms_iou=iou)


st.title("🔍 Deteksi Cacat Pengelasan Menggunakan SSD-ResNet50")
st.write("Aplikasi ini dikembangkan menggunakan SSD-ResNet50 untuk mendeteksi 3 jenis cacat las, yaitu Slag Inclusion, Spatter, dan Undercut.")

tab1, tab2 = st.tabs(["📖 Overview", "🧪 Test"])

with tab1:
    st.subheader("Network Architecture")
    st.image("img/ssdresnet50.png",
             caption="SSD-ResNet50", width="stretch")
    st.markdown(
        "**Penjelasan Arsitektur:**\n"
        "Model ini menggunakan arsitektur Single Shot MultiBox Detector (SSD) dengan backbone ResNet-50 untuk ekstraksi fitur.\n\n"
        "- Backbone ResNet-50 mengekstrak fitur penting dari gambar melalui beberapa lapisan konvolusi dan residual block.\n"
        "- Lapisan fitur tambahan (extra feature layers) digunakan untuk menangkap objek pada berbagai skala.\n"
        "- Detection head melakukan prediksi bounding box dan klasifikasi secara langsung dari fitur tersebut.\n"
        "- Non-Maximum Suppression (NMS) digunakan untuk menghapus prediksi bounding box yang tumpang tindih dan menjaga prediksi dengan confidence tertinggi.\n\n"
        "Kombinasi ini membuat model cepat dan akurat dalam mendeteksi cacat las dengan berbagai ukuran dan bentuk."
    )


with tab2:
    uploaded_file = st.file_uploader(
        "Upload gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.success(f"Berhasil mengunggah: {uploaded_file.name}")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)

        if st.button("🔍 Jalankan Deteksi"):
            with st.spinner("Mendeteksi cacat pengelasan..."):
                model = load_ssd_model(
                    sb.confidence_threshold, sb.iou_threshold)
                result_image, results = model.detect_image(
                    image, return_info=True)

            with col2:
                st.image(result_image, caption="Hasil Deteksi",
                         use_column_width=True)

            with st.expander("📋 Detail Konfigurasi & Hasil Deteksi"):
                st.write(f"- Confidence Threshold: {sb.confidence_threshold}")
                st.write(f"- IoU (NMS) Threshold: {sb.iou_threshold}")
                if results:
                    st.markdown("#### Deteksi:")
                    for idx, det in enumerate(results):
                        xmin, ymin, xmax, ymax, class_id, score = det
                        st.write(
                            f"- **{idx+1}. Kelas:** {model.class_names[class_id]} | **Score:** {score:.2f}")
                        st.write(
                            f"   BBox: [xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}]")
                else:
                    st.warning("Tidak ada deteksi ditemukan.")
    else:
        st.info("Silakan unggah gambar terlebih dahulu.")
#streamlit run app.py --server.enableXsrfProtection false
