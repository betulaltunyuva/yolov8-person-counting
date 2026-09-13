<div align="center">

# 👥 YOLOv8 Person Counting

### Video-Based Person Detection and Counting Application

YOLOv8 Person Counting detects people in uploaded videos, analyzes each processed frame, and presents the results through a simple Gradio interface.

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.0-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

</div>

---

## 📖 About

YOLOv8 Person Counting is a computer vision project developed to demonstrate person detection and video analysis.

The application accepts an uploaded video, processes its frames using a pretrained YOLOv8 model, and generates an annotated output video. Detection is restricted to the `person` class.

The project also contains an experimental color-based categorization method that analyzes clothing colors using the HSV color space.

> This project was developed for educational and internship purposes. It is not intended for surveillance, biometric identification, or real-world demographic analysis.

---

## 📌 Contents

- [Key Features](#-key-features)
- [How It Works](#️-how-it-works)
- [Technologies](#️-technologies)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output](#-output)
- [Important Limitations](#️-important-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## ✨ Key Features

- Detects people in uploaded videos
- Uses a pretrained YOLOv8 model
- Restricts detection to the `person` class
- Draws bounding boxes around detected people
- Displays the number of detected people in each processed frame
- Produces an annotated MP4 output video
- Presents results through a Gradio interface
- Creates a summary table using Pandas
- Uses frame sampling to improve processing performance
- Includes experimental HSV-based color analysis

---

## ⚙️ How It Works

1. The user uploads an MP4 video through the Gradio interface.
2. OpenCV reads and processes the video frames.
3. YOLOv8 detects people in each processed frame.
4. Bounding boxes are drawn around detected people.
5. The upper and lower clothing regions are analyzed in the HSV color space.
6. The processed frames are written to a new video file.
7. The annotated video and a summary table are returned to the user.

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Core application language |
| YOLOv8 | Person detection |
| OpenCV | Video processing and annotation |
| Gradio | Web-based user interface |
| PyTorch | Deep-learning runtime |
| NumPy | Numerical operations |
| Pandas | Summary table creation |

---

## 📁 Project Structure

```text
yolov8-person-counting/
├── images/
├── app.py
├── requirements.txt
└── README.md
```

The `outputs/` directory is created automatically when the application processes a video.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/betulaltunyuva/yolov8-person-counting.git
cd yolov8-person-counting
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
venv\Scripts\activate
```

#### Linux or macOS

```bash
source venv/bin/activate
```

### 4. Install the Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Start the application with:

```bash
python app.py
```

Open the local Gradio address displayed in the terminal. Upload an MP4 video and wait for the analysis to finish.

The pretrained `yolov8n.pt` model may be downloaded automatically during the first run.

---

## 📊 Output

The application returns:

- An annotated output video
- Frame-based category counts
- Percentage values for the displayed categories
- A summary table generated with Pandas

Generated videos are saved inside the automatically created `outputs/` directory.

The summary table represents the results from the final processed frame. The application does not currently track unique individuals throughout the entire video.

---

## ⚠️ Important Limitations

The color-based category estimation used in this project is a simple experimental heuristic.

It uses red, pink, and blue color ratios found in clothing regions. It does not recognize a person’s actual gender and must not be treated as a reliable demographic classification system.

Results may be affected by:

- Clothing colors
- Lighting conditions
- Video quality
- Camera angle
- Partially visible people
- Object-detection errors

The application performs frame-based detection. A person appearing in multiple frames is not counted as one unique individual across the entire video.

---

## 🔮 Future Improvements

- Add multi-object tracking for unique person counting
- Remove or replace the color-based categorization method
- Add configurable confidence thresholds
- Improve video processing speed
- Support additional video formats
- Add performance evaluation metrics
- Add automated tests
- Add GPU configuration guidance

---

## 👩‍💻 Author

**Betül Altunyuva**

Software Engineering Student

[GitHub Profile](https://github.com/betulaltunyuva)

---

<div align="center">

Developed for computer vision learning and experimentation.

⭐ If you find the project useful, consider giving it a star.

</div>
