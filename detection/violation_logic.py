"""
TrafficEye AI - Violation Detection Logic
Rule-based engine to detect traffic violations
"""

import cv2
from datetime import datetime
import os

class ViolationDetector:
    """Detects traffic violations based on rules"""
    
    def __init__(self, output_dir='static/evidence'):
        """
        Initialize violation detector
        Args:
            output_dir: Directory to save evidence images
        """
        self.output_dir = output_dir
        self.stop_line_y = None  # Y-coordinate of stop line (set based on camera angle)
        self.violations_detected = []
        # Keep track of which vehicle tracks have already been fined
        # so that the same car/bike is not fined multiple times for
        # the same violation type in one continuous event.
        # Keys are tuples: (violation_type, track_id)
        self.penalized_track_violations = set()
        # Track previous positions per track to detect actual line crossing
        # events instead of static overlap with the line.
        self.track_last_center_y = {}
        self.track_last_bbox = {}
        # Safety buffer around stop line to reduce jitter-based false positives.
        self.stop_line_buffer_px = 12
        # Minimum movement required between frames to consider crossing valid.
        self.min_cross_motion_px = 3
        # Keep red state briefly to survive per-frame traffic-light misses.
        self.last_red_seen_at = None
        self.red_light_grace_sec = 1.2
        # Require repeated no-helmet observations for the same track
        # before creating a violation (reduces false positives).
        self.no_helmet_streak = {}
        self.no_helmet_required_streak = max(1, int(os.getenv('NO_HELMET_REQUIRED_STREAK', '3')))
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def _is_recent_similar_violation(self, violation_type, bbox, time_window_sec=5, iou_threshold=0.5):
        """Return True if a very similar violation was already recorded recently.

        This is a safety net in case tracking IDs change; it compares the
        position (IoU on bounding boxes) and timestamp of past violations
        so the same bike is not fined again every few frames.
        """
        if not self.violations_detected:
            return False

        x1, y1, x2, y2 = bbox
        area = max((x2 - x1) * (y2 - y1), 1)
        now = datetime.now()

        # Only check a limited number of recent violations for efficiency
        for v in reversed(self.violations_detected[-50:]):
            if v.get('type') != violation_type:
                continue

            v_ts = v.get('timestamp') or now
            try:
                dt = (now - v_ts).total_seconds()
            except Exception:
                dt = time_window_sec + 1

            if dt > time_window_sec:
                # Too old to be the same continuous event
                continue

            vx1, vy1, vx2, vy2 = v.get('bbox', (0, 0, 0, 0))
            inter_x1 = max(x1, vx1)
            inter_y1 = max(y1, vy1)
            inter_x2 = min(x2, vx2)
            inter_y2 = min(y2, vy2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            v_area = max((vx2 - vx1) * (vy2 - vy1), 1)
            union_area = area + v_area - inter_area
            iou = inter_area / max(union_area, 1)

            if iou >= iou_threshold:
                return True

        return False
        
    def set_stop_line(self, y_coordinate):
        """
        Set the stop line position
        Args:
            y_coordinate: Y position of stop line in frame
        """
        self.stop_line_y = y_coordinate
    
    def check_red_light_violation(self, vehicles, traffic_lights, frame):
        """
        Check if any vehicle crossed stop line during red light
        Args:
            vehicles: List of detected vehicles
            traffic_lights: List of detected traffic lights
            frame: Current video frame
        Returns:
            List of violations detected
        """
        violations = []
        
        # Check if any traffic light is red (with short grace window).
        now = datetime.now()
        red_now = any(light.get('status') == 'red' for light in traffic_lights)
        if red_now:
            self.last_red_seen_at = now

        red_light_active = red_now
        if not red_light_active and self.last_red_seen_at is not None:
            dt = (now - self.last_red_seen_at).total_seconds()
            red_light_active = dt <= self.red_light_grace_sec
        
        if not red_light_active or self.stop_line_y is None:
            return violations
        
        # Check each vehicle
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_center_y = int((y1 + y2) / 2)
            track_id = vehicle.get('track_id')
            violation_type = 'Red Light Violation'

            # For reliable red-light enforcement, require a tracked vehicle
            # and detect a true crossing of the stop line between frames.
            if track_id is None:
                continue

            previous_center_y = self.track_last_center_y.get(track_id)
            previous_bbox = self.track_last_bbox.get(track_id)
            self.track_last_center_y[track_id] = vehicle_center_y
            self.track_last_bbox[track_id] = vehicle['bbox']

            if previous_center_y is None or previous_bbox is None:
                continue

            # Direction-aware crossing using the vehicle front edge.
            # This avoids misses where only the nose crosses first and center
            # has not crossed yet.
            dy = vehicle_center_y - previous_center_y
            moved_enough = abs(dy) >= self.min_cross_motion_px
            if not moved_enough:
                continue

            prev_x1, prev_y1, prev_x2, prev_y2 = previous_bbox
            if dy < 0:
                # Moving upward in frame: front edge is top edge (y1)
                prev_front_y = prev_y1
                curr_front_y = y1
                crossed_line = (
                    prev_front_y >= (self.stop_line_y + self.stop_line_buffer_px)
                    and curr_front_y <= (self.stop_line_y - self.stop_line_buffer_px)
                )
            else:
                # Moving downward in frame: front edge is bottom edge (y2)
                prev_front_y = prev_y2
                curr_front_y = y2
                crossed_line = (
                    prev_front_y <= (self.stop_line_y - self.stop_line_buffer_px)
                    and curr_front_y >= (self.stop_line_y + self.stop_line_buffer_px)
                )

            # If vehicle actually crossed the line during red light
            if crossed_line:
                # If we know the tracking id and this track has already
                # been fined for this violation type, skip creating
                # another fine for the same car.
                key = (violation_type, track_id)
                if key in self.penalized_track_violations:
                    continue
                self.penalized_track_violations.add(key)

                violation = {
                    'type': violation_type,
                    'violation_type': violation_type,
                    'vehicle_type': vehicle['type'],
                    'timestamp': datetime.now(),
                    'confidence': vehicle['confidence'],
                    'location': 'Junction A',  # Can be configured
                    'evidence_frame': frame.copy(),
                    'bbox': vehicle['bbox'],
                    'track_id': track_id,
                }
                violations.append(violation)
        
        return violations
    
    def check_helmet_violation(self, motorcycles, frame, helmet_detector):
        """
        Check if motorcycle riders are wearing helmets
        Args:
            motorcycles: List of detected motorcycles
            frame: Current video frame
            helmet_detector: Detector instance with helmet detection capability
        Returns:
            List of violations detected
        """
        violations = []

        for motorcycle in motorcycles:
            if motorcycle['type'] not in ['motorcycle', 'bicycle']:
                continue
            
            # Check for helmet in the upper portion of bounding box
            x1, y1, x2, y2 = motorcycle['bbox']
            
            # Create ROI for head area (upper 40% of bike bounding box)
            head_height = int((y2 - y1) * 0.4)
            head_roi_bbox = (x1, y1, x2, y1 + head_height)
            
            # Check if helmet is present
            has_helmet = helmet_detector.detect_helmet(frame, head_roi_bbox)
            track_id = motorcycle.get('track_id')

            # Need a stable track to accumulate confidence across frames.
            if track_id is None:
                continue

            if has_helmet is True:
                self.no_helmet_streak[track_id] = 0
                continue

            if has_helmet is None:
                # Uncertain prediction: do not escalate to challan.
                continue

            # has_helmet is explicitly False
            current_streak = self.no_helmet_streak.get(track_id, 0) + 1
            self.no_helmet_streak[track_id] = current_streak

            if current_streak < self.no_helmet_required_streak:
                continue
            
            # Fine only on explicit "no helmet" result.
            # If detector returns None (uncertain), skip fining.
            if has_helmet is False:
                violation_type = 'No Helmet'

                # Avoid multiple helmet fines for the same bike/track
                if track_id is not None:
                    key = (violation_type, track_id)
                    if key in self.penalized_track_violations:
                        continue
                    self.penalized_track_violations.add(key)

                # Also check for very similar recent violation in
                # almost the same position (helps when track IDs
                # change but it is still the same physical bike).
                if self._is_recent_similar_violation(violation_type, motorcycle['bbox']):
                    continue

                violation = {
                    'type': violation_type,
                    'violation_type': violation_type,
                    'vehicle_type': 'motorcycle',
                    'timestamp': datetime.now(),
                    'confidence': motorcycle['confidence'],
                    'location': 'Junction A',
                    'evidence_frame': frame.copy(),
                    'bbox': motorcycle['bbox'],
                    'track_id': track_id,
                }
                violations.append(violation)
        
        return violations
    
    def save_evidence(self, violation):
        """
        Save violation evidence to disk
        Args:
            violation: Violation dictionary
        Returns:
            Path to saved evidence image
        """
        timestamp = violation['timestamp'].strftime('%Y%m%d_%H%M%S')
        filename = f"{violation['type'].replace(' ', '_')}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Draw violation box on frame
        frame = violation['evidence_frame'].copy()
        x1, y1, x2, y2 = violation['bbox']
        
        # Draw red box around violating vehicle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Add violation text
        violation_text = f"{violation['type']} - {violation['vehicle_type']}"
        time_text = violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Add semi-transparent overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Add text
        cv2.putText(frame, violation_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, time_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save image
        cv2.imwrite(filepath, frame)
        
        return filename
    
    def process_frame(self, frame, vehicles, traffic_lights, detector):
        """
        Process a single frame for all violations
        Args:
            frame: Video frame
            vehicles: Detected vehicles
            traffic_lights: Detected traffic lights
            detector: Main detector instance
        Returns:
            List of all violations found
        """
        all_violations = []
        
        # Check red light violations
        red_light_violations = self.check_red_light_violation(
            vehicles, traffic_lights, frame
        )
        all_violations.extend(red_light_violations)
        
        # Check helmet violations
        motorcycles = [v for v in vehicles if v['type'] == 'motorcycle']
        helmet_violations = self.check_helmet_violation(
            motorcycles, frame, detector
        )
        all_violations.extend(helmet_violations)
        
        # Save evidence for each violation
        for violation in all_violations:
            evidence_path = self.save_evidence(violation)
            violation['evidence_image'] = evidence_path
            self.violations_detected.append(violation)
        
        return all_violations
    
    def draw_stop_line(self, frame):
        """Draw stop line on frame for visualization"""
        if self.stop_line_y:
            cv2.line(frame, (0, self.stop_line_y), 
                    (frame.shape[1], self.stop_line_y), (0, 0, 255), 3)
            cv2.putText(frame, "STOP LINE", (10, self.stop_line_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

if __name__ == "__main__":
    print("✅ Violation Logic Module Ready")
