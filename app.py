import cv2
import uuid
import os
import time
import gradio as gr
from ultralytics import YOLO
import pandas as pd

# ================== Ayarlar (videona göre oynatılabilir) ==================
RESIZE_DIV        = 6          # videoyu 1/6 ölçekte işle (hız)
FRAME_STEP        = 2          # her 2 kareden birini işle (hız)
# ==========================================================================

model = YOLO("yolov8n.pt")
 # istersen 'best.pt'
print("Model Sınıfları:", model.names)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video açılamadı.")
        return None, None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  // RESIZE_DIV
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // RESIZE_DIV
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25

    os.makedirs("outputs", exist_ok=True)
    output_filename = f"output_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join("outputs", output_filename)

    # Windows için en sorunsuz codec
    # --- Writer'ı güvenli kur: önce H.264 (avc1), olmazsa mp4v, en sonda XVID ---
    def make_writer(path, fps, width, height):
        for fc in ["avc1", "H264", "mp4v", "XVID"]:
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fc), fps, (width, height))
            if w.isOpened():
                print(f"[VideoWriter] codec -> {fc}")
                return w
            else:
                w.release()
        return None

    # Kareler atlandığı için FPS de bölünür, yoksa video hızlı oynar.
    out = make_writer(output_path, fps / FRAME_STEP, width, height)
    if out is None:
        print("[HATA] VideoWriter açılamadı (codec bulunamadı).")
        return None, None

    # Her video için yeni model: takipçi önceki videodan ID taşımasın.
    tracker_model = YOLO("yolov8n.pt")

    frame_count = 0
    unique_ids = set()   # videoda görülen farklı kişilerin takip ID'leri
    max_in_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_STEP != 0:
            frame_count += 1
            continue

        resized_frame = cv2.resize(frame, (width, height))

        # --- YOLO takip: sadece 'person', ID'ler kareler arasında korunur ---
        results = tracker_model.track(
            source=resized_frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.25,
            iou=0.40,
            imgsz=640,
            verbose=False
        )[0]

        # print(f"Kare {frame_count}: YOLO kutu = {len(results.boxes)}")

        # Debug: kaç kutu çıktı?
        # print("boxes:", len(results.boxes))

        boxes = results.boxes
        track_ids = (
            boxes.id.int().tolist() if boxes.id is not None
            else [None] * len(boxes)
        )

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if track_id is None:
                # Takipçi bu kişiye henüz ID vermedi
                label_text = "?"
                color = (0, 255, 255)
            else:
                unique_ids.add(track_id)
                label_text = f"ID {track_id}"
                color = (0, 200, 0)

            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resized_frame, label_text, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        max_in_frame = max(max_in_frame, len(boxes))

        info = f"Kare: {frame_count} | Karede: {len(boxes)} | Toplam: {len(unique_ids)}"
        if frame_count % 10 == 0:
            print(info)

        cv2.putText(resized_frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(resized_frame)
        frame_count += 1

    cap.release()
    out.release()
    time.sleep(0.5)

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        return None, None

    return output_path, len(unique_ids), max_in_frame

# --- Gradio: dosya yolunu çıkarmak için yardımcı ---
def _as_path(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("name") or v.get("path")
    if hasattr(v, "name"):
        return v.name
    return None

def handle_video(video):
    video_path = _as_path(video)  # gr.File'dan gelen nesneden yol çıkar
    if not video_path or not os.path.exists(video_path):
        return None, None

    result_path, unique_count, max_in_frame = process_video(video_path)
    if result_path is None:
        return None, None

    table = pd.DataFrame({
        "Ölçüm": ["Videodaki farklı kişi sayısı", "Bir karede en fazla kişi"],
        "Sayı": [unique_count, max_in_frame]
    })

    # Sıra: (Video, DataFrame)
    return result_path, table

demo = gr.Interface(
    fn=handle_video,
    inputs=gr.Video(label="Video Yükle (.mp4)", sources=["upload"], format="mp4"),
    outputs=[
        gr.Video(label="İşlenmiş Video", autoplay=True, loop=True),
        gr.Dataframe(label="Kişi Sayımı", interactive=False)
    ],
    title="YOLOv8 Kişi Sayımı",
    description="YOLO ile kişi tespiti, ByteTrack ile takip. Her kişi videoda bir kez sayılır."
)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        debug=True,
        show_error=True,
        # share=True  # proxy engeli varsa bunu aç
    )
