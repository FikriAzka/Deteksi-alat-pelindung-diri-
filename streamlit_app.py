import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

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

    # -------- VIDEO --------
    else:
        cap = cv2.VideoCapture(temp_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25

        target_width = 640
        target_height = 480

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(out_path, fourcc, fps, (target_width, target_height))

        st.info("⏳ Memproses video...")
        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 🔥 resize FIXED
            frame = cv2.resize(frame, (target_width, target_height))

            results = model(frame, conf=0.4, verbose=False)[0]
            annotated = results.plot()

            # ⚠️ WAJIB ukuran sama
            annotated = cv2.resize(annotated, (target_width, target_height))
            out.write(annotated)

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()
        progress.empty()

        st.success("✅ Video siap diputar")
        st.video(out_path)



