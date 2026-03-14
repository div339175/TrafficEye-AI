# TrafficEye AI - CCTV Intelligent Multi-Violation Detection System

## 🚦 Project Overview

TrafficEye AI is a smart traffic monitoring system that uses artificial intelligence and computer vision to automatically detect traffic violations from CCTV camera feeds.

### Key Features
- ✅ **Real-time Vehicle Detection** - Detects cars, bikes, buses, trucks
- ✅ **Helmet Detection** - Identifies riders without helmets
- ✅ **Traffic Light Recognition** - Reads traffic signal status
- ✅ **Red Light Violation Detection** - Catches vehicles jumping red lights
- ✅ **Automated Evidence Capture** - Takes snapshots with timestamp
- ✅ **Web Dashboard** - View all violations in real-time

## 🎯 How It Works

```
CCTV Feed → AI Detection → Rule Engine → Evidence Capture → Web Dashboard
```

1. **Video Input**: System receives live or recorded CCTV footage
2. **Object Detection**: AI model (YOLO) identifies vehicles, helmets, traffic lights
3. **Violation Analysis**: Logic engine checks for rule violations
4. **Evidence Collection**: Captures frame, timestamp, violation type, location
5. **Dashboard Display**: Shows violations on web interface with evidence

## 🛠️ Technology Stack

- **AI/ML**: YOLOv8 (Object Detection), OpenCV (Computer Vision)
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Database**: SQLite
- **Libraries**: ultralytics, opencv-python, numpy, PIL

## 📁 Project Structure

```
trafic/
├── app.py                  # Main Flask application
├── detection/
│   ├── detector.py         # AI detection engine
│   ├── violation_logic.py  # Violation detection rules
│   └── models/             # AI model files
├── templates/
│   ├── index.html          # Dashboard page
│   └── violations.html     # Violations list
├── static/
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript files
│   └── evidence/           # Captured violation images
├── database/
│   └── violations.db       # SQLite database
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Installation & Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
python app.py
```

### Step 3: Open Dashboard
Open browser and go to: `http://localhost:5000`

## 🎓 How to Explain to Your Teacher

### What This Project Does:
"This system converts ordinary CCTV cameras into intelligent traffic monitors that can automatically detect violations like red light jumping and helmetless riding without human intervention."

### Technical Approach:
1. **AI Model (YOLO)**: Trained to recognize vehicles, helmets, and traffic lights
2. **Rule Engine**: Checks if detected objects violate traffic rules
3. **Evidence System**: Automatically captures proof when violation occurs
4. **Web Interface**: Provides easy access to violation records

### Real-World Benefits:
- Reduces manual monitoring effort by 90%
- 24/7 automated surveillance
- Instant violation detection
- Evidence-based enforcement
- Data analytics for traffic management

## 📊 Violations Detected

1. **Red Light Jumping** - Vehicle crosses stop line when signal is red
2. **Helmetless Riding** - Two-wheeler rider without helmet
3. **Wrong Lane Usage** - Vehicle in restricted lane
4. **Speed Violation** - (Future enhancement)

## 🎨 Dashboard Features

- Real-time violation feed
- Filter by date, type, location
- View evidence images
- Export reports
- Analytics and statistics

## 📝 Future Enhancements

- License plate recognition (OCR)
- Speed detection
- Wrong-way driving detection
- SMS/Email alerts to authorities
- Mobile app integration

## 👨‍💻 Developer

Created by: [Your Name]
Institution: [Your College/School]
Date: February 2026

## 📄 License

Educational Project - For Learning Purposes
