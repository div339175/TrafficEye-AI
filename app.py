"""
TrafficEye AI - Main Flask Application
Author: Divakar Maurya
Description: Web interface for traffic violation detection system
             with e-Challan, ANPR, and AI-powered analytics
"""

from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename
import os
import json
import threading
import time
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database', 'violations.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'evidence')
app.config['VIDEO_UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Initialize database
db = SQLAlchemy(app)

# ─────────────────────────────────────────────
# Global Detection State
# ─────────────────────────────────────────────
detection_state = {
    'running': False,
    'thread': None,
    'progress': 0,
    'total_frames': 0,
    'processed_frames': 0,
    'status': 'idle',        # idle, initializing, processing, completed, error
    'message': '',
    'violations_found': 0,
    'current_frame': None,   # Latest processed frame (JPEG bytes) for streaming
    'video_file': None,
    'stop_flag': False,
}


# ─────────────────────────────────────────────
# Database Models
# ─────────────────────────────────────────────

class Violation(db.Model):
    """Database model for storing traffic violations"""
    id = db.Column(db.Integer, primary_key=True)
    violation_type = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    location = db.Column(db.String(200))
    vehicle_type = db.Column(db.String(50))
    evidence_image = db.Column(db.String(300))
    confidence = db.Column(db.Float)
    status = db.Column(db.String(50), default='Pending')
    # New fields for ANPR & innovations
    plate_number = db.Column(db.String(20), default='')
    vehicle_color = db.Column(db.String(30), default='')
    speed_kmh = db.Column(db.Float, default=0.0)
    severity_score = db.Column(db.Integer, default=0)
    severity_level = db.Column(db.String(20), default='')
    day_night = db.Column(db.String(10), default='day')

    def to_dict(self):
        return {
            'id': self.id,
            'violation_type': self.violation_type,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'location': self.location,
            'vehicle_type': self.vehicle_type,
            'evidence_image': self.evidence_image,
            'confidence': self.confidence,
            'status': self.status,
            'plate_number': self.plate_number or 'Not Detected',
            'vehicle_color': self.vehicle_color or 'Unknown',
            'speed_kmh': self.speed_kmh or 0.0,
            'severity_score': self.severity_score or 0,
            'severity_level': self.severity_level or 'LOW',
            'day_night': self.day_night or 'day',
        }


class Challan(db.Model):
    """Database model for e-Challans"""
    id = db.Column(db.Integer, primary_key=True)
    challan_number = db.Column(db.String(50), unique=True, nullable=False)
    violation_id = db.Column(db.Integer, db.ForeignKey('violation.id'))
    payment_ref = db.Column(db.String(30))
    fine_amount = db.Column(db.Integer, default=500)
    late_fee = db.Column(db.Integer, default=1000)
    penalty_points = db.Column(db.Integer, default=0)
    violation_section = db.Column(db.String(100))
    violation_description = db.Column(db.String(200))
    plate_number = db.Column(db.String(20), default='')
    vehicle_type = db.Column(db.String(50))
    location = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    due_date = db.Column(db.DateTime)
    status = db.Column(db.String(30), default='Pending')
    state = db.Column(db.String(50))
    challan_image = db.Column(db.String(300))
    evidence_image = db.Column(db.String(300))

    def to_dict(self):
        return {
            'id': self.id,
            'challan_number': self.challan_number,
            'violation_id': self.violation_id,
            'payment_ref': self.payment_ref,
            'fine_amount': self.fine_amount,
            'late_fee': self.late_fee,
            'penalty_points': self.penalty_points,
            'violation_section': self.violation_section,
            'violation_description': self.violation_description,
            'plate_number': self.plate_number or 'Not Detected',
            'vehicle_type': self.vehicle_type,
            'location': self.location,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp else '',
            'due_date': self.due_date.strftime('%Y-%m-%d') if self.due_date else '',
            'status': self.status,
            'state': self.state or 'Unknown',
            'challan_image': self.challan_image,
            'evidence_image': self.evidence_image,
        }


# Create database tables and directories
with app.app_context():
    os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static', 'evidence'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static', 'challans'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static', 'uploads'), exist_ok=True)
    db.create_all()


# ─────────────────────────────────────────────
# Detection Engine (runs in background thread)
# ─────────────────────────────────────────────

def run_detection_pipeline(video_path):
    """
    Run the full detection pipeline on an uploaded video.
    Runs in a background thread — saves violations & challans to DB,
    streams processed frames to the frontend via MJPEG.
    """
    global detection_state

    detection_state['status'] = 'initializing'
    detection_state['message'] = 'Loading AI models...'
    detection_state['violations_found'] = 0
    detection_state['stop_flag'] = False

    try:
        # Torch 2.6 compatibility for older YOLO checkpoints
        os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

        # Import all detection modules
        from detection.detector import TrafficDetector
        from detection.violation_logic import ViolationDetector
        from detection.anpr import NumberPlateRecognizer
        from detection.echallan import EChallanGenerator
        from detection.innovations import (
            SpeedEstimator, SeverityScorer, DayNightDetector,
            VehicleColorDetector, TripleRidingDetector,
            PeakHourAnalyzer, ViolationHeatmap
        )

        # Initialize all modules
        traffic_detector = TrafficDetector()
        violation_detector = ViolationDetector()
        anpr = NumberPlateRecognizer()
        challan_gen = EChallanGenerator()
        speed_estimator = SpeedEstimator()
        severity_scorer = SeverityScorer()
        day_night_detector = DayNightDetector()
        color_detector = VehicleColorDetector()
        triple_detector = TripleRidingDetector()
        peak_analyzer = PeakHourAnalyzer()

        detection_state['message'] = 'Opening video file...'

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            detection_state['status'] = 'error'
            detection_state['message'] = f'Cannot open video: {video_path}'
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        detection_state['total_frames'] = total_frames

        # Read first frame to set up
        ret, frame = cap.read()
        if not ret:
            detection_state['status'] = 'error'
            detection_state['message'] = 'Cannot read video frames'
            cap.release()
            return

        height, width = frame.shape[:2]
        stop_line_y = int(height * 0.6)
        violation_detector.set_stop_line(stop_line_y)
        heatmap = ViolationHeatmap(width, height)

        # Reset to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        detection_state['status'] = 'processing'
        detection_state['message'] = 'Analyzing video for violations...'

        frame_count = 0
        process_every_n = 3  # Process every 3rd frame for speed

        with app.app_context():
            while True:
                if detection_state['stop_flag']:
                    detection_state['message'] = 'Detection stopped by user'
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                detection_state['processed_frames'] = frame_count
                detection_state['progress'] = int((frame_count / max(total_frames, 1)) * 100)

                if frame_count % process_every_n != 0:
                    # Still encode the raw frame for streaming (lower CPU)
                    display_frame = frame.copy()
                    # Draw stop line on every frame
                    cv2.line(display_frame, (0, stop_line_y), (width, stop_line_y), (0, 255, 255), 2)
                    cv2.putText(display_frame, 'STOP LINE', (10, stop_line_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    _, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    detection_state['current_frame'] = jpeg.tobytes()
                    continue

                # ── Full analysis on every Nth frame ──

                # Day/Night detection
                lighting = day_night_detector.detect_mode(frame)
                enhanced_frame = day_night_detector.enhance_night_frame(frame)

                # Detect vehicles
                vehicles = traffic_detector.detect_vehicles(enhanced_frame)

                # Estimate speed
                vehicles = speed_estimator.update(vehicles, frame_count)

                # Detect traffic lights
                traffic_lights = traffic_detector.detect_traffic_light(enhanced_frame)

                # Check for red light / helmet violations
                violations = violation_detector.process_frame(
                    enhanced_frame, vehicles, traffic_lights, traffic_detector
                )

                # Check overspeeding
                for vehicle in vehicles:
                    if vehicle.get('is_overspeeding') and vehicle.get('speed_kmh', 0) > 40:
                        violation = {
                            'type': 'Overspeeding',
                            'vehicle_type': vehicle['type'],
                            'timestamp': datetime.now(),
                            'confidence': vehicle['confidence'],
                            'location': 'Junction A',
                            'evidence_frame': enhanced_frame.copy(),
                            'bbox': vehicle['bbox'],
                            'speed_kmh': vehicle['speed_kmh'],
                        }
                        evidence_path = violation_detector.save_evidence(violation)
                        violation['evidence_image'] = evidence_path
                        violations.append(violation)

                # Check triple riding
                for vehicle in vehicles:
                    if vehicle['type'] == 'motorcycle':
                        all_objects = traffic_detector.detect_objects(enhanced_frame)
                        detected_persons = []
                        for result in all_objects:
                            for box in result.boxes:
                                cls_name = result.names[int(box.cls[0])]
                                if cls_name == 'person':
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    detected_persons.append({
                                        'type': 'person',
                                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                                    })
                        triple_result = triple_detector.check_triple_riding(
                            enhanced_frame, vehicle['bbox'], detected_persons
                        )
                        if triple_result['is_triple_riding']:
                            violation = {
                                'type': 'Triple Riding',
                                'vehicle_type': 'motorcycle',
                                'timestamp': datetime.now(),
                                'confidence': vehicle['confidence'],
                                'location': 'Junction A',
                                'evidence_frame': enhanced_frame.copy(),
                                'bbox': vehicle['bbox'],
                            }
                            evidence_path = violation_detector.save_evidence(violation)
                            violation['evidence_image'] = evidence_path
                            violations.append(violation)

                # ── Save each violation to DB + generate e-Challan ──
                for violation in violations:
                    # ANPR
                    plate_result = anpr.recognize_plate(enhanced_frame, violation.get('bbox'))
                    plate_number = plate_result['text'] if plate_result else ''

                    # Vehicle color
                    color_result = color_detector.detect_color(enhanced_frame, violation['bbox'])

                    # Severity
                    severity = severity_scorer.calculate_severity(violation)

                    # Heatmap
                    cx = (violation['bbox'][0] + violation['bbox'][2]) // 2
                    cy = (violation['bbox'][1] + violation['bbox'][3]) // 2
                    heatmap.add_violation(cx, cy)
                    peak_analyzer.log_violation(violation['type'])

                    # Save violation to DB
                    db_violation = Violation(
                        violation_type=violation['type'],
                        vehicle_type=violation['vehicle_type'],
                        location=violation.get('location', 'Junction A'),
                        evidence_image=violation.get('evidence_image', ''),
                        confidence=violation.get('confidence', 0.0),
                        plate_number=plate_number,
                        vehicle_color=color_result['color'],
                        speed_kmh=violation.get('speed_kmh', 0.0),
                        severity_score=severity['score'],
                        severity_level=severity['level'],
                        day_night=lighting,
                    )
                    db.session.add(db_violation)
                    db.session.commit()

                    # Generate e-Challan
                    challan_data = challan_gen.generate_challan({
                        'violation_type': violation['type'],
                        'vehicle_type': violation['vehicle_type'],
                        'plate_number': plate_number,
                        'location': violation.get('location', 'Junction A'),
                        'evidence_image': violation.get('evidence_image', ''),
                        'confidence': violation.get('confidence', 0.0),
                        'timestamp': violation.get('timestamp', datetime.now()),
                    })

                    db_challan = Challan(
                        challan_number=challan_data['challan_number'],
                        violation_id=db_violation.id,
                        payment_ref=challan_data['payment_ref'],
                        fine_amount=challan_data['fine_amount'],
                        late_fee=challan_data['late_fee'],
                        penalty_points=challan_data['penalty_points'],
                        violation_section=challan_data['violation_section'],
                        violation_description=challan_data['violation_description'],
                        plate_number=plate_number,
                        vehicle_type=violation['vehicle_type'],
                        location=violation.get('location', 'Junction A'),
                        due_date=datetime.strptime(challan_data['due_date'], '%Y-%m-%d'),
                        status='Pending',
                        state=challan_data.get('state', 'Unknown'),
                        challan_image=challan_data.get('challan_image', ''),
                        evidence_image=violation.get('evidence_image', ''),
                    )
                    db.session.add(db_challan)
                    db.session.commit()

                    detection_state['violations_found'] += 1

                # ── Draw annotated frame for live streaming ──
                display_frame = traffic_detector.draw_detections(frame, vehicles, traffic_lights)
                display_frame = violation_detector.draw_stop_line(display_frame)
                display_frame = speed_estimator.draw_speed(display_frame, vehicles)

                # Overlay stats
                overlay_y = 30
                cv2.putText(display_frame, f"Vehicles: {len(vehicles)}", (10, overlay_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                overlay_y += 30
                cv2.putText(display_frame, f"Mode: {lighting.upper()}", (10, overlay_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                overlay_y += 30
                cv2.putText(display_frame, f"Violations: {detection_state['violations_found']}", (10, overlay_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                overlay_y += 30
                cv2.putText(display_frame, f"Progress: {detection_state['progress']}%", (10, overlay_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Encode frame for MJPEG streaming
                _, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                detection_state['current_frame'] = jpeg.tobytes()

                # Update status message
                detection_state['message'] = (
                    f'Processing frame {frame_count}/{total_frames} '
                    f'({detection_state["progress"]}%) — '
                    f'{detection_state["violations_found"]} violations found'
                )

        cap.release()

        # Save heatmap
        try:
            heatmap.save_heatmap()
        except Exception:
            pass

        if not detection_state['stop_flag']:
            detection_state['status'] = 'completed'
            detection_state['progress'] = 100
            detection_state['message'] = (
                f'✅ Detection complete! Processed {frame_count} frames. '
                f'Found {detection_state["violations_found"]} violations.'
            )
        else:
            detection_state['status'] = 'completed'

    except Exception as e:
        detection_state['status'] = 'error'
        detection_state['message'] = f'Error: {str(e)}'
        print(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detection_state['running'] = False


# ─────────────────────────────────────────────
# Page Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/violations')
def violations_page():
    return render_template('violations.html')

@app.route('/challans')
def challans_page():
    return render_template('challans.html')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')


# ─────────────────────────────────────────────
# Violations API
# ─────────────────────────────────────────────

@app.route('/api/violations', methods=['GET'])
def get_violations():
    violation_type = request.args.get('type', None)
    if violation_type:
        violations = Violation.query.filter_by(violation_type=violation_type).order_by(Violation.timestamp.desc()).all()
    else:
        violations = Violation.query.order_by(Violation.timestamp.desc()).limit(100).all()
    return jsonify([v.to_dict() for v in violations])

@app.route('/api/violations/<int:violation_id>', methods=['GET'])
def get_violation(violation_id):
    violation = Violation.query.get_or_404(violation_id)
    return jsonify(violation.to_dict())


# ─────────────────────────────────────────────
# Stats API (enhanced)
# ─────────────────────────────────────────────

@app.route('/api/stats', methods=['GET'])
def get_stats():
    total_violations = Violation.query.count()
    red_light_count = Violation.query.filter_by(violation_type='Red Light Violation').count()
    no_helmet_count = Violation.query.filter_by(violation_type='No Helmet').count()
    overspeeding_count = Violation.query.filter_by(violation_type='Overspeeding').count()
    triple_riding_count = Violation.query.filter_by(violation_type='Triple Riding').count()

    plates_detected = Violation.query.filter(
        Violation.plate_number != '',
        Violation.plate_number != 'Not Detected',
        Violation.plate_number.isnot(None)
    ).count()

    total_challans = Challan.query.count()
    total_fines = db.session.query(db.func.sum(Challan.fine_amount)).scalar() or 0
    pending_challans = Challan.query.filter_by(status='Pending').count()
    critical_count = Violation.query.filter(Violation.severity_level == 'CRITICAL').count()

    return jsonify({
        'total': total_violations,
        'red_light': red_light_count,
        'no_helmet': no_helmet_count,
        'overspeeding': overspeeding_count,
        'triple_riding': triple_riding_count,
        'plates_detected': plates_detected,
        'total_challans': total_challans,
        'total_fines': total_fines,
        'pending_challans': pending_challans,
        'critical_violations': critical_count,
        'today': Violation.query.filter(
            db.func.date(Violation.timestamp) == datetime.utcnow().date()
        ).count(),
    })


# ─────────────────────────────────────────────
# e-Challan API
# ─────────────────────────────────────────────

@app.route('/api/challans', methods=['GET'])
def get_challans():
    status_filter = request.args.get('status', None)
    if status_filter:
        challans = Challan.query.filter_by(status=status_filter).order_by(Challan.timestamp.desc()).all()
    else:
        challans = Challan.query.order_by(Challan.timestamp.desc()).limit(100).all()
    return jsonify([c.to_dict() for c in challans])

@app.route('/api/challans/<int:challan_id>', methods=['GET'])
def get_challan(challan_id):
    challan = Challan.query.get_or_404(challan_id)
    return jsonify(challan.to_dict())

@app.route('/api/challans/<int:challan_id>/pay', methods=['POST'])
def pay_challan(challan_id):
    challan = Challan.query.get_or_404(challan_id)
    challan.status = 'Paid'
    db.session.commit()
    return jsonify({'status': 'success', 'message': f'Challan {challan.challan_number} marked as Paid'})

@app.route('/api/challan_stats', methods=['GET'])
def get_challan_stats():
    total = Challan.query.count()
    pending = Challan.query.filter_by(status='Pending').count()
    paid = Challan.query.filter_by(status='Paid').count()
    total_collected = db.session.query(db.func.sum(Challan.fine_amount)).filter_by(status='Paid').scalar() or 0
    total_pending_amount = db.session.query(db.func.sum(Challan.fine_amount)).filter_by(status='Pending').scalar() or 0
    return jsonify({
        'total': total, 'pending': pending, 'paid': paid,
        'total_collected': total_collected,
        'total_pending_amount': total_pending_amount,
    })


# ─────────────────────────────────────────────
# Analytics API
# ─────────────────────────────────────────────

@app.route('/api/analytics/hourly', methods=['GET'])
def get_hourly_analytics():
    violations = Violation.query.all()
    hourly = {h: 0 for h in range(24)}
    for v in violations:
        if v.timestamp:
            hourly[v.timestamp.hour] = hourly.get(v.timestamp.hour, 0) + 1
    return jsonify(hourly)

@app.route('/api/analytics/severity', methods=['GET'])
def get_severity_analytics():
    result = {}
    for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        result[level] = Violation.query.filter_by(severity_level=level).count()
    return jsonify(result)

@app.route('/api/analytics/plates', methods=['GET'])
def get_plate_analytics():
    violations = Violation.query.filter(
        Violation.plate_number != '',
        Violation.plate_number != 'Not Detected',
        Violation.plate_number.isnot(None)
    ).order_by(Violation.timestamp.desc()).limit(20).all()
    return jsonify([{
        'plate': v.plate_number, 'vehicle_type': v.vehicle_type,
        'color': v.vehicle_color, 'violation': v.violation_type,
        'timestamp': v.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
    } for v in violations])

@app.route('/api/analytics/type_distribution', methods=['GET'])
def get_type_distribution():
    types = db.session.query(
        Violation.violation_type, db.func.count(Violation.id)
    ).group_by(Violation.violation_type).all()
    return jsonify({t: c for t, c in types})


# ─────────────────────────────────────────────
# Video Upload & Detection Control API
# ─────────────────────────────────────────────

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Upload a video file and start detection automatically"""
    global detection_state

    if detection_state['running']:
        return jsonify({'status': 'error', 'message': 'Detection is already running. Stop it first.'}), 400

    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Save uploaded video
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_name = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], save_name)
    file.save(filepath)

    # Start detection in background thread
    detection_state['running'] = True
    detection_state['video_file'] = filepath
    detection_state['progress'] = 0
    detection_state['processed_frames'] = 0
    detection_state['current_frame'] = None
    detection_state['stop_flag'] = False

    thread = threading.Thread(target=run_detection_pipeline, args=(filepath,), daemon=True)
    detection_state['thread'] = thread
    thread.start()

    return jsonify({
        'status': 'started',
        'message': f'Video "{filename}" uploaded. Detection started!',
        'filename': save_name,
    })


@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    """Start detection on a pre-existing video source"""
    global detection_state

    if detection_state['running']:
        return jsonify({'status': 'error', 'message': 'Detection already running'}), 400

    data = request.json or {}
    source = data.get('source', 'demo_video.mp4')

    # Resolve the video path
    if source.isdigit():
        video_path = int(source)  # webcam index
    else:
        video_path = os.path.join(BASE_DIR, source)
        if not os.path.exists(video_path):
            # Try static/uploads
            video_path = os.path.join(BASE_DIR, 'static', 'uploads', source)
        if not os.path.exists(str(video_path)):
            return jsonify({'status': 'error', 'message': f'Video file not found: {source}'}), 404

    detection_state['running'] = True
    detection_state['video_file'] = video_path
    detection_state['progress'] = 0
    detection_state['processed_frames'] = 0
    detection_state['current_frame'] = None
    detection_state['stop_flag'] = False

    thread = threading.Thread(target=run_detection_pipeline, args=(video_path,), daemon=True)
    detection_state['thread'] = thread
    thread.start()

    return jsonify({'status': 'started', 'message': f'Detection started on: {source}'})


@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    """Stop running detection"""
    global detection_state

    if not detection_state['running']:
        return jsonify({'status': 'info', 'message': 'No detection running'})

    detection_state['stop_flag'] = True
    return jsonify({'status': 'stopped', 'message': 'Stopping detection...'})


@app.route('/api/detection_status', methods=['GET'])
def detection_status():
    """Get current detection progress and status"""
    return jsonify({
        'running': detection_state['running'],
        'status': detection_state['status'],
        'progress': detection_state['progress'],
        'total_frames': detection_state['total_frames'],
        'processed_frames': detection_state['processed_frames'],
        'violations_found': detection_state['violations_found'],
        'message': detection_state['message'],
    })


@app.route('/api/video_feed')
def video_feed():
    """MJPEG stream of the detection processing (live video in browser)"""
    def generate():
        while detection_state['running'] or detection_state['current_frame']:
            frame = detection_state.get('current_frame')
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # Send a blank frame if no detection frame yet
                time.sleep(0.1)
                continue
            time.sleep(0.03)  # ~30fps max

            # If detection completed and we've sent the last frame, stop
            if not detection_state['running'] and detection_state['status'] == 'completed':
                # Send the last frame a few more times then break
                for _ in range(10):
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.1)
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/clear_data', methods=['POST'])
def clear_data():
    """Clear all violations and challans (reset for new detection)"""
    with app.app_context():
        Challan.query.delete()
        Violation.query.delete()
        db.session.commit()
    return jsonify({'status': 'success', 'message': 'All data cleared'})


@app.route('/static/challans/<path:filename>')
def serve_challan(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static', 'challans'), filename)


if __name__ == '__main__':
    print("=" * 60)
    print("🚦 TrafficEye AI - Starting Application")
    print("=" * 60)
    print("📍 Dashboard:  http://localhost:8080")
    print("📋 Violations: http://localhost:8080/violations")
    print("🧾 e-Challans: http://localhost:8080/challans")
    print("📊 Analytics:  http://localhost:8080/analytics")
    print("🔍 Detection Engine: Ready")
    print("=" * 60)
    app.run(debug=False, use_reloader=False, threaded=True, host='0.0.0.0', port=8080)
