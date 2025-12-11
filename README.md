# 🎯 AI Assistive Object Detection System for the Visually Impaired

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLO](https://img.shields.io/badge/YOLO-v5%2Fv8-green.svg)](https://github.com/ultralytics/yolov5)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An intelligent real-time object detection and hazard alert system designed to enhance mobility, confidence, and safety for visually impaired individuals through AI-powered computer vision and intuitive audio feedback.

---

## 📖 Overview

This AI-powered assistive system provides real-time object detection and hazard alerts to help visually impaired individuals navigate their environment safely and independently. By combining state-of-the-art YOLO deep learning models with distance estimation, voice alerts, and intelligent video buffering, the system delivers actionable situational awareness through auditory feedback.

### Key Capabilities

- **Real-time Object Detection**: Identifies obstacles, people, vehicles, and environmental objects using YOLO
- **Distance Estimation**: Calculates approximate distances to detected objects
- **Intelligent Hazard Analysis**: Categorizes risks as low, medium, or high priority
- **Voice Guidance**: Provides clear, actionable spoken instructions
- **Incident Recording**: Automatically saves video clips during hazardous events
- **Edge Device Ready**: Optimized for deployment on lightweight hardware like Raspberry Pi

---

## 🏗️ System Architecture

The software follows a modular pipeline architecture:

```
┌─────────────────────┐
│  Frame Acquisition  │ ──► Camera/IP Stream Input
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ YOLO Object Detect  │ ──► Bounding Boxes + Confidence Scores
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Distance Estimation │ ──► Calculate Object Distance
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Hazard Analysis    │ ──► Risk Classification
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Audio Alert Engine │ ──► Voice Instructions
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Video Buffering    │ ──► Incident Clip Storage
└─────────────────────┘
```

---

## 🧩 Software Components

### 1. **Frame Capture Module**
- Handles camera input via OpenCV
- Supports webcam, USB cameras, IP cameras, and prerecorded videos
- Configurable resolution and frame rate

### 2. **YOLO Detection Module**
- Utilizes YOLOv5/YOLOv8 pretrained or custom-trained models
- Performs real-time object detection (20-30 FPS depending on hardware)
- Multi-class object recognition

### 3. **Distance Estimation Engine**
- Calculates distance from bounding box dimensions
- Optional integration with ultrasonic/depth sensors
- Calibrated for accuracy within navigation context

### 4. **Hazard Detection System**
- Evaluates detection confidence and distance metrics
- Applies configurable risk thresholds
- Prioritizes immediate threats

### 5. **Audio Output System**
- Text-to-Speech (TTS) using pyttsx3 for offline operation
- Provides clear, context-aware navigation guidance:
  - _"Obstacle ahead"_
  - _"Move slightly to the left"_
  - _"Vehicle approaching"_
- Adjustable speech rate and volume

### 6. **Video Buffer Manager**
- Maintains rolling buffer of recent frames (5-10 seconds)
- Efficient memory management
- Triggered recording during hazard events

### 7. **Incident Recorder**
- Automatically saves video clips during danger events
- Timestamped recordings for review and analysis
- Configurable storage duration

---

## ⚙️ Requirements

### Software Requirements

**Programming Language:**
- Python 3.8 or higher

**Core Libraries:**
```
opencv-python>=4.5.0
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
pyttsx3>=2.90
ultralytics  # For YOLOv8
```

**Optional Libraries:**
- `tensorrt` / `onnxruntime` - For accelerated inference
- `RPi.GPIO` - For Raspberry Pi sensor integration
- `pyaudio` - For advanced audio processing

### Hardware Requirements

**Minimum:**
- CPU: Dual-core processor (ARM/x86)
- RAM: 2GB
- Camera: USB webcam or CSI camera module

**Recommended:**
- CPU: Quad-core processor
- GPU: NVIDIA GPU with CUDA support (for faster inference)
- RAM: 4GB+
- Camera: HD webcam (720p/1080p)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Aadithya2222/object-detection-system-for-the-visually-impaired.git
cd object-detection-system-for-the-visually-impaired
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLO Weights
```bash
# For YOLOv5
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

# Or use custom trained weights
# Place your custom weights in the weights/ directory
```

---

## 🎮 Usage

### Basic Usage (Webcam)
```bash
python main.py --model gpModel.pt --source 0
```

### Using IP Camera
```bash
python main.py --model gpModel.pt --source "http://192.168.1.100:8080/video"
```

### Using Video File
```bash
python main.py --model gpModel.pt --source path/to/video.mp4
```

### Advanced Options
```bash
python main.py \
  --model gpModel.pt \
  --source 0 \
  --conf-threshold 0.5 \
  --distance-threshold 2.0 \
  --enable-audio \
  --save-incidents
```

---

## 🧪 Testing & Simulation

The system supports multiple testing modes:

### 1. **Real-time Camera Testing**
Test with live camera feed for immediate feedback

### 2. **Video-based Simulation**
Use prerecorded videos to test detection accuracy

### 3. **Synthetic Data Testing**
Inject simulated sensor data for edge case validation

### 4. **Audio Output Verification**
Test TTS clarity and timing

### 5. **Incident Replay**
Review saved clips to evaluate system performance

---

## 📊 Features

| Feature | Description |
|---------|-------------|
| ✅ Real-time Detection | YOLO-based object detection at 20-30 FPS |
| ✅ Multi-class Recognition | People, vehicles, obstacles, animals, etc. |
| ✅ Distance Estimation | Geometric calculation from bounding boxes |
| ✅ Hazard Classification | Low/Medium/High risk categorization |
| ✅ Voice Alerts | Immediate auditory warnings and guidance |
| ✅ Rolling Buffer | Short-term video memory for context |
| ✅ Incident Recording | Automatic clip saving during hazards |
| ✅ Edge Optimized | Runs on Raspberry Pi and similar devices |
| ✅ Offline Operation | No internet required for core functionality |

---

## 📦 Output

The system generates:

1. **Visual Output** (optional display):
   - Bounding boxes around detected objects
   - Confidence scores and labels
   - Distance indicators

2. **Audio Output**:
   - Context-aware voice instructions
   - Priority-based alert timing
   - Adjustable verbosity

3. **Saved Media**:
   - Timestamped incident video clips
   - Detection logs with metadata
   - Event summaries

4. **Logs**:
   ```
   [2025-12-11 10:23:45] HAZARD: Vehicle detected at 1.2m - HIGH RISK
   [2025-12-11 10:23:47] ACTION: Audio alert triggered
   [2025-12-11 10:23:48] SAVE: Incident clip saved to incidents/20251211_102345.mp4
   ```

---

## 🎯 Use Cases

- **Indoor Navigation**: Detecting furniture, doorways, and obstacles
- **Outdoor Mobility**: Identifying vehicles, pedestrians, and street hazards
- **Public Transport**: Navigating stations and platforms safely
- **Emergency Situations**: Quick hazard identification and avoidance
- **Daily Activities**: Shopping, commuting, and social interactions

---

## 🛠️ Configuration

Edit configuration settings in `config.yaml` or pass as command-line arguments:

```yaml
# Detection Settings
confidence_threshold: 0.5
distance_threshold: 2.0  # meters
hazard_levels:
  low: 3.0
  medium: 2.0
  high: 1.0

# Audio Settings
audio_enabled: true
speech_rate: 150
volume: 0.8

# Recording Settings
buffer_duration: 10  # seconds
save_incidents: true
incident_path: "incidents/"
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Feature enhancements
- Documentation improvements
- Testing and validation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Aadithya** - [@Aadithya2222](https://github.com/Aadithya2222)

---

## 🙏 Acknowledgments

- **YOLO**: Ultralytics for the YOLO framework
- **OpenCV**: Open Source Computer Vision Library
- **PyTorch**: Deep learning framework
- Community contributors and testers

---

## 📞 Support

For questions, issues, or feedback:
- Open an issue on GitHub
- Contact: [Create an issue](https://github.com/Aadithya2222/object-detection-system-for-the-visually-impaired/issues)

---

## 🌟 Future Enhancements

- [ ] Integration with GPS for location-aware alerts
- [ ] Mobile app interface
- [ ] Cloud-based incident analysis
- [ ] Multi-language TTS support
- [ ] Advanced depth sensor integration (LiDAR)
- [ ] Smart glasses compatibility
- [ ] Indoor mapping and navigation
- [ ] Crowd density detection

---

<div align="center">

**Making the world more accessible, one detection at a time.** 🌍

_By offering real-time auditory feedback about approaching objects, this application redefines possibilities for independent mobility, contributing to a more inclusive and accessible world for the visually impaired._

</div>







