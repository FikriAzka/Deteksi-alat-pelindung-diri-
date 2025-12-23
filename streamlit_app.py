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

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        st.info("⏳ Memproses video, mohon tunggu...")
        progress = st.progress(0)

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 🔥 RESIZE supaya cepat
            frame = cv2.resize(frame, (640, int(640 * height / width)))

            results = model(frame, conf=0.4, verbose=False)[0]
            annotated = results.plot()

            # resize balik biar ukuran video konsisten
            annotated = cv2.resize(annotated, (width, height))
            out.write(annotated)

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()
        progress.empty()

        st.success("✅ Video selesai diproses")
        st.video(out_path)


