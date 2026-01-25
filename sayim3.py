import cv2
import uuid
import os
import time
import gradio as gr
from ultralytics import YOLO
import numpy as np
import pandas as pd

# ================== Ayarlar (videona göre oynatılabilir) ==================
RESIZE_DIV        = 6          # videoyu 1/2 ölçekte işle (hız)
MIN_BOX_WH        = 32         # çok küçük kutuları at
SAT_THRESH        = 60         # doygunluk eşiği (S>?)
VAL_THRESH        = 50         # parlaklık eşiği (V>?)
RED_RATIO_THRESH  = 0.18       # kırmızı/pembe oran eşiği -> Kadın
BLUE_RATIO_THRESH = 0.18       # mavi oran eşiği -> Erkek
UPPER_PORTION     = 0.55       # kutunun üst % kaçını analiz edelim
# ==========================================================================

model = YOLO("yolov8n.pt")
 # istersen 'best.pt'
print("Model Sınıfları:", model.names)

def classify_gender_by_color(person_crop, debug=False):
    if person_crop.size == 0:
        return "?"

    h, w = person_crop.shape[:2]
    if h < MIN_BOX_WH or w < MIN_BOX_WH:
        return "?"

    upper = person_crop[: int(h * UPPER_PORTION), :]
    lower = person_crop[int(h * 0.55):, :]

    # --- gevşetilmiş eşikler (kadın tarafını aç) ---
    SAT = max(45, SAT_THRESH)
    VAL = max(40, VAL_THRESH)

    # Daha az agresif ten maskesi (kırmızı kıyafetleri silmesin)
    def hue_stats_skip_skin(img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        skin = (Cr >= 133) & (Cr <= 177) & (Cb >= 77) & (Cb <= 127)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        sat_val = (S > SAT) & (V > VAL)

        mask = sat_val & (~skin)
        if mask.sum() == 0:
            return 0.0, 0.0

        Hm = H[mask]
        red_ratio  = ((Hm <= 20).sum() + (Hm >= 150).sum()) / Hm.size   # kırmızı [0..20] U [150..179]
        blue_ratio = ((Hm >= 85) & (Hm <= 150)).sum() / Hm.size         # mavi [85..150]
        return float(red_ratio), float(blue_ratio)

    red_u,  blue_u  = hue_stats_skip_skin(upper)
    red_l,  blue_l  = hue_stats_skip_skin(lower)

    # Ağırlıklar: kadın sinyali üstte, erkek sinyali altta
    red_total  = 0.8*red_u  + 0.2*red_l
    blue_total = 0.3*blue_u + 0.7*blue_l

    MARGIN   = 0.04
    RED_MIN  = 0.12
    BLUE_MIN = 0.12

    if debug:
        print(f"red_u={red_u:.2f} red_l={red_l:.2f} -> red_total={red_total:.2f} | "
              f"blue_u={blue_u:.2f} blue_l={blue_l:.2f} -> blue_total={blue_total:.2f}")

    if red_total > blue_total + MARGIN and red_total > RED_MIN:
        return "Kadin"
    if blue_total > red_total + MARGIN and blue_total > BLUE_MIN:
        return "Erkek"
    return "?"

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video açılamadı.")
        return None, None, None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  // RESIZE_DIV
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // RESIZE_DIV
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    os.makedirs("outputs", exist_ok=True)
    output_filename = f"output_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join("outputs", output_filename)

    # Windows için en sorunsuz codec
    # --- Writer'ı güvenli kur: önce H.264 (avc1), olmazsa mp4v, en sonda XVID ---
    def make_writer(path, fps, width, height):
        for fc in ["avc1", "H264", "mp4v", "XVID"]:
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fc), fps or 24, (width, height))
            if w.isOpened():
                print(f"[VideoWriter] codec -> {fc}")
                return w
            else:
                w.release()
        return None

    out = make_writer(output_path, fps, width, height)
    if out is None:
        print("[HATA] VideoWriter açılamadı (codec bulunamadı).")
        return None, None, None

    frame_count = 0
    last_female_count = 0
    last_male_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 2 != 0:
            frame_count += 1
            continue

        resized_frame = cv2.resize(frame, (width, height))

        # --- YOLO tahmin: sadece 'person', daha düşük conf, sabit imgsz ---
        # --- YOLO tahmin: kutu sayısını kontrol et ---
        results = model.predict(
            source=resized_frame,
            classes=None,
            conf=0.25,
            iou=0.40,
            imgsz=640,
            verbose=False
        )[0]

        # print(f"Kare {frame_count}: YOLO kutu = {len(results.boxes)}")

        # Debug: kaç kutu çıktı?
        # print("boxes:", len(results.boxes))

        female_count = 0
        male_count   = 0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(resized_frame.shape[1]-1, x2)
            y2 = min(resized_frame.shape[0]-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = resized_frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (128, 256))
            tag = classify_gender_by_color(crop)

            if tag == "Kadin":
                female_count += 1
                label_text = f"Kadin{female_count}"
                color = (0, 0, 255)
            elif tag == "Erkek":
                male_count += 1
                label_text = f"Erkek{male_count}"
                color = (255, 0, 0)
            else:
                label_text = "?"
                color = (0, 255, 255)

            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resized_frame, label_text, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        last_female_count = female_count
        last_male_count = male_count

        info = f"Kare: {frame_count} | Kadın: {female_count} | Erkek: {male_count}"
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
        return None, None, None

    return output_path, last_female_count, last_male_count

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

    result_path, female_count, male_count = process_video(video_path)
    if result_path is None:
        return None, None

    toplam = max(1, female_count + male_count)
    k_pct = 100.0 * female_count / toplam
    e_pct = 100.0 * male_count / toplam

    table = pd.DataFrame({
        "Tür": ["Kadın", "Erkek"],
        "Sayı": [female_count, male_count],
        "Yüzde": [f"{k_pct:.1f}%", f"{e_pct:.1f}%"]
    })

    # Sıra: (Video, DataFrame)
    return result_path, table

demo = gr.Interface(
    fn=handle_video,
    inputs=gr.Video(label="Video Yükle (.mp4)", sources=["upload"], format="mp4"),
    outputs=[
        gr.Video(label="İşlenmiş Video", autoplay=True, loop=True),
        gr.Dataframe(label="Toplam İnsan Sayısı", interactive=False)
    ],
    title="Kadın ve Erkek Sayımı (HSV üst gövde analizi)",
    description="YOLO ile kişi, HSV ile üst gövdeden renk analizi. Gri/düşük doygunlukta saymaz."
)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        debug=True,
        show_error=True,
        # share=True  # proxy engeli varsa bunu aç
    )
