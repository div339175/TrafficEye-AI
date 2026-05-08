"""
TrafficEye AI - Innovative Features Module
Unique features that differentiate from other traffic systems:

1. 🚗 Vehicle Speed Estimation (using frame-to-frame tracking)
2. 🌡️ Violation Heatmap Generation
3. 🕐 Smart Peak Hour Analysis
4. 🔔 Real-time Alert Scoring (AI Severity Engine)
5. 🌙 Day/Night Mode Auto-Detection
6. 👥 Triple Riding Detection on Two-Wheelers
7. 📊 Predictive Analytics (violation trend prediction)
8. 🏍️ Vehicle Color Detection (for identification)
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os
import math


class SpeedEstimator:
    """
    Estimate vehicle speed using optical flow between consecutive frames.
    Innovation: Most systems only detect static violations. This estimates
    actual speed from CCTV without any road sensors.
    """

    def __init__(self, fps=30, pixels_per_meter=8.0):
        """
        Args:
            fps: Video frames per second
            pixels_per_meter: Calibration factor (pixels in frame = 1 meter)
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.prev_positions = {}  # Track vehicle positions: id -> deque of positions
        self.vehicle_id_counter = 0
        self.speed_limit = 40  # km/h (city default)

    def update(self, vehicles, frame_count):
        """
        Track vehicles across frames and estimate speed.

        Args:
            vehicles: List of detected vehicles with 'bbox'
            frame_count: Current frame number

        Returns:
            List of vehicles with added 'speed_kmh' and 'is_overspeeding'
        """
        current_positions = {}

        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Try to match with existing tracked vehicle
            matched_id = self._match_vehicle(center_x, center_y)

            if matched_id is None:
                # New vehicle
                self.vehicle_id_counter += 1
                matched_id = self.vehicle_id_counter
                self.prev_positions[matched_id] = deque(maxlen=10)

            self.prev_positions[matched_id].append({
                'x': center_x, 'y': center_y,
                'frame': frame_count
            })

            current_positions[matched_id] = True

            # Calculate speed if we have enough history
            speed = self._calculate_speed(matched_id)
            vehicle['speed_kmh'] = round(speed, 1)
            vehicle['is_overspeeding'] = speed > self.speed_limit
            vehicle['track_id'] = matched_id

        # Clean up lost tracks
        lost_ids = [k for k in self.prev_positions if k not in current_positions]
        for lost_id in lost_ids:
            if len(self.prev_positions[lost_id]) > 0:
                last = self.prev_positions[lost_id][-1]
                if frame_count - last['frame'] > self.fps:  # Lost for > 1 second
                    del self.prev_positions[lost_id]

        return vehicles

    def _match_vehicle(self, cx, cy, max_distance=80):
        """Match current detection to existing tracked vehicle"""
        best_id = None
        best_dist = max_distance

        for vid, positions in self.prev_positions.items():
            if not positions:
                continue
            last = positions[-1]
            dist = math.sqrt((cx - last['x'])**2 + (cy - last['y'])**2)
            if dist < best_dist:
                best_dist = dist
                best_id = vid

        return best_id

    def _calculate_speed(self, vehicle_id):
        """Calculate speed from position history"""
        positions = self.prev_positions.get(vehicle_id, deque())
        if len(positions) < 2:
            return 0.0

        # Use first and last positions
        p1 = positions[0]
        p2 = positions[-1]

        pixel_distance = math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
        frame_diff = p2['frame'] - p1['frame']

        if frame_diff == 0:
            return 0.0

        # Convert to real-world speed
        meter_distance = pixel_distance / self.pixels_per_meter
        time_seconds = frame_diff / self.fps
        speed_ms = meter_distance / max(time_seconds, 0.001)
        speed_kmh = speed_ms * 3.6

        # Clamp unreasonable values
        return min(speed_kmh, 200)

    def set_speed_limit(self, limit_kmh):
        """Set the speed limit for this zone"""
        self.speed_limit = limit_kmh

    def draw_speed(self, frame, vehicles):
        """Draw speed information on frame"""
        output = frame.copy()
        for v in vehicles:
            if 'speed_kmh' not in v:
                continue
            x1, y1, x2, y2 = v['bbox']
            speed = v['speed_kmh']
            color = (0, 0, 255) if v.get('is_overspeeding') else (0, 200, 0)

            label = f"{speed:.0f} km/h"
            cv2.putText(output, label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if v.get('is_overspeeding'):
                cv2.putText(output, "OVERSPEEDING!", (x1, y2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return output


class ViolationHeatmap:
    """
    Generate spatial heatmap of where violations occur most.
    Innovation: Visual heatmap overlay showing violation hotspots.
    """

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.heatmap_data = np.zeros((height, width), dtype=np.float32)
        self.violation_points = []

    def add_violation(self, x, y, intensity=1.0):
        """Add a violation point to the heatmap"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.violation_points.append({'x': x, 'y': y, 'intensity': intensity})
            # Add gaussian blob at violation point
            self._add_gaussian(x, y, intensity)

    def _add_gaussian(self, cx, cy, intensity, sigma=30):
        """Add a gaussian blob to heatmap"""
        for y in range(max(0, cy - sigma * 3), min(self.height, cy + sigma * 3)):
            for x in range(max(0, cx - sigma * 3), min(self.width, cx + sigma * 3)):
                dist_sq = (x - cx)**2 + (y - cy)**2
                val = intensity * math.exp(-dist_sq / (2 * sigma**2))
                self.heatmap_data[y, x] += val

    def generate_overlay(self, frame):
        """Generate heatmap overlay on frame"""
        if frame.shape[:2] != (self.height, self.width):
            self.height, self.width = frame.shape[:2]
            # Resize heatmap data if needed
            if self.heatmap_data.shape != (self.height, self.width):
                self.heatmap_data = cv2.resize(
                    self.heatmap_data, (self.width, self.height))

        # Normalize heatmap
        if self.heatmap_data.max() > 0:
            normalized = (self.heatmap_data / self.heatmap_data.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros((self.height, self.width), dtype=np.uint8)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Blend with original frame
        alpha = 0.4
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay

    def get_hotspot_zones(self, top_n=5):
        """Get the top N violation hotspot zones"""
        # Find peaks in heatmap
        kernel_size = 50
        blurred = cv2.GaussianBlur(self.heatmap_data, (kernel_size * 2 + 1, kernel_size * 2 + 1), 0)

        hotspots = []
        temp = blurred.copy()
        for _ in range(top_n):
            if temp.max() == 0:
                break
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp)
            hotspots.append({
                'x': int(max_loc[0]),
                'y': int(max_loc[1]),
                'intensity': float(max_val)
            })
            # Zero out this area
            cv2.circle(temp, max_loc, kernel_size * 2, 0, -1)

        return hotspots

    def save_heatmap(self, output_path='static/evidence/heatmap.jpg'):
        """Save heatmap as standalone image"""
        if self.heatmap_data.max() > 0:
            normalized = (self.heatmap_data / self.heatmap_data.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros((self.height, self.width), dtype=np.uint8)

        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, colored)
        return output_path


class PeakHourAnalyzer:
    """
    Analyze violations by time of day to find peak violation hours.
    Innovation: Helps traffic police allocate resources to the right hours.
    """

    def __init__(self):
        self.hourly_counts = defaultdict(int)
        self.daily_counts = defaultdict(int)
        self.type_by_hour = defaultdict(lambda: defaultdict(int))
        self.violations_log = []

    def log_violation(self, violation_type, timestamp=None):
        """Log a violation with timestamp"""
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            try:
                timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                timestamp = datetime.now()

        hour = timestamp.hour
        day = timestamp.strftime('%A')

        self.hourly_counts[hour] += 1
        self.daily_counts[day] += 1
        self.type_by_hour[hour][violation_type] += 1
        self.violations_log.append({
            'type': violation_type,
            'timestamp': timestamp.isoformat(),
            'hour': hour,
            'day': day,
        })

    def get_peak_hours(self, top_n=3):
        """Get top N peak violation hours"""
        sorted_hours = sorted(self.hourly_counts.items(),
                              key=lambda x: x[1], reverse=True)
        return [{'hour': h, 'count': c,
                 'label': f"{h:02d}:00 - {h + 1:02d}:00"}
                for h, c in sorted_hours[:top_n]]

    def get_hourly_distribution(self):
        """Get violation count for each hour (0-23)"""
        return {h: self.hourly_counts.get(h, 0) for h in range(24)}

    def get_daily_distribution(self):
        """Get violation count by day of week"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                'Friday', 'Saturday', 'Sunday']
        return {d: self.daily_counts.get(d, 0) for d in days}

    def get_risk_score(self, hour=None):
        """
        Get risk score (0-100) for current or specified hour.
        Higher = more violations expected.
        """
        if hour is None:
            hour = datetime.now().hour
        if not self.hourly_counts:
            return 50  # Default medium risk
        max_count = max(self.hourly_counts.values()) if self.hourly_counts else 1
        current_count = self.hourly_counts.get(hour, 0)
        return int((current_count / max(max_count, 1)) * 100)


class SeverityScorer:
    """
    AI-based violation severity scoring.
    Innovation: Not just detect violations, but score them by danger level.
    """

    SEVERITY_WEIGHTS = {
        'Red Light Violation': 0.9,
        'No Helmet': 0.8,
        'Overspeeding': 0.95,
        'Triple Riding': 0.7,
        'Wrong Lane': 0.6,
        'No Seatbelt': 0.5,
        'Using Mobile Phone': 0.85,
    }

    VEHICLE_RISK = {
        'motorcycle': 1.3,  # Higher risk
        'bicycle': 1.2,
        'car': 1.0,
        'bus': 1.5,  # Large vehicle = more damage potential
        'truck': 1.5,
    }

    def __init__(self):
        self.scores_history = []

    def calculate_severity(self, violation):
        """
        Calculate severity score (0-100) for a violation.

        Factors:
        - Violation type danger level
        - Vehicle type risk
        - Time of day (night = higher risk)
        - AI confidence
        - Speed (if available)
        """
        base_weight = self.SEVERITY_WEIGHTS.get(
            violation.get('violation_type', ''), 0.5)

        vehicle_risk = self.VEHICLE_RISK.get(
            violation.get('vehicle_type', 'car'), 1.0)

        # Time factor: night (10pm-6am) is riskier
        timestamp = violation.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                timestamp = datetime.now()
        hour = timestamp.hour
        time_factor = 1.3 if (hour >= 22 or hour <= 6) else 1.0

        # Confidence factor
        confidence = violation.get('confidence', 0.5)
        conf_factor = 0.5 + (confidence * 0.5)  # Range: 0.5-1.0

        # Speed factor
        speed = violation.get('speed_kmh', 0)
        speed_factor = 1.0
        if speed > 60:
            speed_factor = 1.5
        elif speed > 40:
            speed_factor = 1.2

        # Calculate final score
        raw_score = base_weight * vehicle_risk * time_factor * conf_factor * speed_factor
        severity = min(int(raw_score * 100), 100)

        # Classify
        if severity >= 80:
            level = 'CRITICAL'
            color = '#dc2626'
        elif severity >= 60:
            level = 'HIGH'
            color = '#ea580c'
        elif severity >= 40:
            level = 'MEDIUM'
            color = '#ca8a04'
        else:
            level = 'LOW'
            color = '#16a34a'

        result = {
            'score': severity,
            'level': level,
            'color': color,
            'factors': {
                'violation_danger': base_weight,
                'vehicle_risk': vehicle_risk,
                'time_factor': time_factor,
                'confidence_factor': conf_factor,
                'speed_factor': speed_factor,
            }
        }

        self.scores_history.append(result)
        return result


class DayNightDetector:
    """
    Auto-detect if the CCTV feed is in day or night mode.
    Innovation: Adapts detection thresholds based on lighting condition.
    """

    def __init__(self):
        self.current_mode = 'day'
        self.brightness_history = deque(maxlen=30)

    def detect_mode(self, frame):
        """
        Detect if frame is day or night based on brightness.

        Returns: 'day', 'night', or 'twilight'
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        self.brightness_history.append(avg_brightness)

        # Use rolling average for stability
        avg = np.mean(list(self.brightness_history))

        if avg > 120:
            self.current_mode = 'day'
        elif avg > 60:
            self.current_mode = 'twilight'
        else:
            self.current_mode = 'night'

        return self.current_mode

    def get_adjusted_confidence(self, base_confidence):
        """Adjust confidence threshold based on lighting"""
        if self.current_mode == 'night':
            return max(base_confidence - 0.1, 0.3)  # Lower threshold at night
        elif self.current_mode == 'twilight':
            return max(base_confidence - 0.05, 0.35)
        return base_confidence

    def enhance_night_frame(self, frame):
        """Enhance frame for better detection at night"""
        if self.current_mode != 'night':
            return frame

        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced


class VehicleColorDetector:
    """
    Detect dominant color of vehicles for better identification.
    Innovation: Helps narrow down vehicle identity even without plate.
    """

    COLOR_RANGES = {
        'Red': ([0, 100, 100], [10, 255, 255]),
        'Red2': ([160, 100, 100], [180, 255, 255]),
        'Blue': ([100, 100, 100], [130, 255, 255]),
        'Green': ([40, 100, 100], [80, 255, 255]),
        'Yellow': ([20, 100, 100], [40, 255, 255]),
        'White': ([0, 0, 180], [180, 30, 255]),
        'Black': ([0, 0, 0], [180, 255, 50]),
        'Silver': ([0, 0, 120], [180, 30, 180]),
    }

    def detect_color(self, frame, vehicle_bbox):
        """
        Detect dominant color of vehicle.

        Args:
            frame: Full video frame
            vehicle_bbox: (x1, y1, x2, y2)

        Returns:
            dict with 'color', 'confidence'
        """
        x1, y1, x2, y2 = vehicle_bbox

        # Use center portion of vehicle (avoid background)
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        roi = frame[y1 + margin_y:y2 - margin_y, x1 + margin_x:x2 - margin_x]

        if roi.size == 0:
            return {'color': 'Unknown', 'confidence': 0.0}

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = roi.shape[0] * roi.shape[1]

        color_scores = {}
        for color_name, (lower, upper) in self.COLOR_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            count = cv2.countNonZero(mask)
            color_scores[color_name] = count / max(total_pixels, 1)

        # Merge Red ranges
        if 'Red2' in color_scores:
            color_scores['Red'] = color_scores.get('Red', 0) + color_scores.pop('Red2')

        # Get dominant color
        if color_scores:
            best_color = max(color_scores, key=color_scores.get)
            best_score = color_scores[best_color]
            if best_score > 0.15:  # At least 15% of vehicle area
                return {'color': best_color, 'confidence': round(best_score, 2)}

        return {'color': 'Unknown', 'confidence': 0.0}


class TripleRidingDetector:
    """
    Detect if more than 2 people are riding on a two-wheeler.
    Innovation: Specific to Indian traffic violation - very common problem.
    """

    def __init__(self, person_model=None):
        self.person_model = person_model

    def _iou(self, box1, box2):
        """Compute IoU between two bboxes (x1,y1,x2,y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = max((box1[2] - box1[0]) * (box1[3] - box1[1]), 1)
        a2 = max((box2[2] - box2[0]) * (box2[3] - box2[1]), 1)
        return inter / max(a1 + a2 - inter, 1)

    def _dedupe_person_boxes(self, person_boxes, iou_threshold=0.6):
        """Remove duplicate person detections (same rider detected multiple times)."""
        if not person_boxes:
            return []

        # Sort by area descending and keep non-overlapping representatives.
        boxes = sorted(
            person_boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )
        kept = []
        for b in boxes:
            if any(self._iou(b, k) >= iou_threshold for k in kept):
                continue
            kept.append(b)
        return kept

    def check_triple_riding(self, frame, motorcycle_bbox, detected_objects):
        """
        Check if there are more than 2 people on a motorcycle.

        Args:
            frame: Video frame
            motorcycle_bbox: (x1, y1, x2, y2) of motorcycle
            detected_objects: All detected objects in frame

        Returns:
            dict with 'is_triple_riding', 'person_count'
        """
        mx1, my1, mx2, my2 = motorcycle_bbox

        # Expand motorcycle region slightly upward (riders are above the bike)
        expanded_y1 = max(0, my1 - int((my2 - my1) * 0.5))
        expanded_bbox = (mx1, expanded_y1, mx2, my2)

        # Count persons overlapping with motorcycle region
        candidate_person_boxes = []
        for obj in detected_objects:
            if obj.get('type') == 'person':
                px1, py1, px2, py2 = obj['bbox']
                # Check overlap
                if self._bbox_overlap(expanded_bbox, (px1, py1, px2, py2)) > 0.3:
                    candidate_person_boxes.append((px1, py1, px2, py2))

        unique_person_boxes = self._dedupe_person_boxes(candidate_person_boxes)
        person_count = len(unique_person_boxes)

        return {
            'is_triple_riding': person_count >= 3,
            'person_count': person_count,
        }

    def _bbox_overlap(self, box1, box2):
        """Calculate overlap ratio between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / max(area2, 1)


class PredictiveAnalytics:
    """
    Simple trend-based prediction for violations.
    Innovation: Predict when and where violations are likely to increase.
    """

    def __init__(self):
        self.daily_history = defaultdict(list)  # date -> [count, ...]

    def add_daily_count(self, date_str, count):
        """Add daily violation count"""
        self.daily_history[date_str].append(count)

    def predict_next_day(self):
        """Predict tomorrow's violation count based on trend"""
        if len(self.daily_history) < 3:
            return {'predicted_count': 0, 'trend': 'insufficient_data'}

        # Simple moving average prediction
        recent_counts = []
        sorted_dates = sorted(self.daily_history.keys())
        for date_key in sorted_dates[-7:]:
            total = sum(self.daily_history[date_key])
            recent_counts.append(total)

        if len(recent_counts) < 2:
            return {'predicted_count': 0, 'trend': 'insufficient_data'}

        avg = np.mean(recent_counts)
        trend = recent_counts[-1] - recent_counts[0]

        if trend > 0:
            trend_label = 'increasing'
        elif trend < 0:
            trend_label = 'decreasing'
        else:
            trend_label = 'stable'

        predicted = max(0, int(avg + trend * 0.3))

        return {
            'predicted_count': predicted,
            'trend': trend_label,
            'confidence': min(len(recent_counts) / 7.0, 1.0),
            'avg_daily': round(avg, 1),
        }

    def get_weekly_summary(self):
        """Get summary of last 7 days"""
        sorted_dates = sorted(self.daily_history.keys())
        last_7 = sorted_dates[-7:]

        summary = []
        for date_key in last_7:
            total = sum(self.daily_history[date_key])
            summary.append({'date': date_key, 'count': total})

        return summary


if __name__ == "__main__":
    print("Testing Innovation Modules...")

    # Test Speed Estimator
    se = SpeedEstimator()
    print("✅ Speed Estimator ready")

    # Test Severity Scorer
    scorer = SeverityScorer()
    result = scorer.calculate_severity({
        'violation_type': 'Red Light Violation',
        'vehicle_type': 'motorcycle',
        'confidence': 0.9,
    })
    print(f"✅ Severity Score: {result['score']} ({result['level']})")

    # Test Day/Night Detector
    dnd = DayNightDetector()
    print("✅ Day/Night Detector ready")

    # Test Vehicle Color Detector
    vcd = VehicleColorDetector()
    print("✅ Vehicle Color Detector ready")

    # Test Heatmap
    hm = ViolationHeatmap()
    print("✅ Violation Heatmap ready")

    print("\n✅ All innovation modules initialized!")
