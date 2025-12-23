import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO

# ================= LOAD MODEL =================
model = YOLO("best.pt")

# ================= DETECTION FUNCTION =================
def detect(file):
    if file is None:
        return None, None

    path = file.name.lower()

    # -------- IMAGE --------
    if path.endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(file.name)
        results = model(img)[0]
        return results.plot(), None

    # -------- VIDEO --------
    cap = cv2.VideoCapture(file.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 25

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        out.write(results.plot())

    cap.release()
    out.release()

    return None, out_path


# ================= GRADIO APP =================
with gr.Blocks() as demo:
    gr.Markdown("# 🦺 Deteksi APD (Helmet & Safety Vest)")
    gr.Markdown(
        "Upload **gambar atau video** untuk melakukan deteksi APD menggunakan **YOLO**."
    )

    file_in = gr.File(
        label="Upload Gambar / Video",
        file_types=[".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"]
    )

    btn = gr.Button("Deteksi")

    img_out = gr.Image(label="Hasil Deteksi (Gambar)")
    vid_out = gr.Video(label="Hasil Deteksi (Video)")

    btn.click(
        detect,
        inputs=file_in,
        outputs=[img_out, vid_out]
    )

# ⚠️ WAJIB untuk Render
demo.launch(
    server_name="0.0.0.0",
    server_port=10000
)
