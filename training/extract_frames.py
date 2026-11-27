"""Step 1 of the dataset pipeline: sample frames from the training videos.

The extracted frames were uploaded to Roboflow, labeled by hand with the
single class `person`, and exported in YOLO format (see `dataset/`).
"""

import cv2
from pathlib import Path

# Proje ana klasörü
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Eğitim videolarının olduğu klasör (videolar boyut nedeniyle repoda yok)
VIDEO_DIR = PROJECT_DIR / "videos" / "train"

# Çıkarılan görüntülerin kaydedileceği klasör
OUTPUT_DIR = PROJECT_DIR / "frames_raw"

# Her videodan yaklaşık kaç görüntü alınacak
TARGET_FRAMES = 80


def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Video açılamadı: {video_path.name}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Videonun tamamına yayılmış şekilde kare seç
    frame_interval = max(total_frames // TARGET_FRAMES, 1)

    frame_number = 0
    saved_count = 0

    video_name = video_path.stem

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0 and saved_count < TARGET_FRAMES:
            output_name = f"{video_name}_frame_{saved_count:03d}.jpg"
            output_path = OUTPUT_DIR / output_name

            success_write, encoded_image = cv2.imencode(".jpg", frame)

            if success_write:
                encoded_image.tofile(str(output_path))
                saved_count += 1

        frame_number += 1

    cap.release()

    print(f"{video_path.name} -> {saved_count} görüntü çıkarıldı.")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sadece train klasöründeki videolar işlenir
    videos = list(VIDEO_DIR.glob("*.mp4"))

    print(f"{len(videos)} eğitim videosu bulundu.\n")

    for video in videos:
        extract_frames(video)

    print("\nKare çıkarma işlemi tamamlandı.")
    print(f"Görüntüler burada: {OUTPUT_DIR}")
