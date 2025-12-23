import os
import streamlit as st
import tempfile
from ultralytics import YOLO
import cv2
st.write("OpenCV OK:", cv2.__version__)

st.set_page_config(page_title="Deteksi APD", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🦺 Deteksi APD (Helmet & Safety Vest)")
st.write("Upload **gambar atau video** untuk melakukan deteksi APD menggunakan YOLO.")

uploaded_file = st.file_uploader(
    "Upload Gambar / Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    suffix = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # -------- IMAGE --------
    if suffix.lower() in ["jpg", "jpeg", "png"]:
        img = cv2.imread(temp_path)
        results = model(img)[0]
        st.image(results.plot(), caption="Hasil Deteksi", use_container_width=True)

    # ... (bagian upload dan image di atas tetap sama) ...

    # -------- VIDEO --------
    else:
        cap = cv2.VideoCapture(temp_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25

        # --- UBAH DISINI: AMBIL UKURAN ASLI VIDEO ---
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Simpan file sementara hasil OpenCV
        tfile_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path_raw = tfile_output.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path_raw, fourcc, fps, (width, height))

        st.info(f"⏳ Memproses video ({width}x{height})...") # Info resolusi
        progress = st.progress(0)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

       while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Deteksi (Tanpa resize, sesuai request sebelumnya)
            results = model(frame, conf=0.4, verbose=False)[0]

            # 2. GAMBAR OTOMATIS (Ini pengganti loop manual tadi)
            # .plot() akan mengembalikan frame baru yang sudah ada kotak-kotaknya
            annotated_frame = results.plot() 

            # 3. Simpan ke video
            out.write(annotated_frame) 

            frame_count += 1
            if total_frames > 0:
                progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()
        progress.empty()

        # --- KONVERSI FFmpeg (Wajib agar bisa diputar di web) ---
        output_path_fixed = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        st.write("🔄 Melakukan encoding video...")
        
        # Note: Proses ini mungkin sedikit lebih lama jika resolusi video besar
        import os
        os.system(f"ffmpeg -i {output_path_raw} -vcodec libx264 {output_path_fixed} -y")

        st.success("✅ Video siap diputar")
        st.video(output_path_fixed)









