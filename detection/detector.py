"""
TrafficEye AI - Object Detection Module
Uses YOLOv8 for vehicle, helmet, and traffic light detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

class TrafficDetector:
    """Main detection class for TrafficEye AI"""
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the detector
        Args:
            model_path: Path to YOLO model file
        """
        print("🔧 Initializing TrafficEye AI Detector...")
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"⚠️  Using default YOLOv8 nano model")
            self.model = YOLO('yolov8n.pt')  # Nano model (lightweight)
        
        # Define classes we're interested in
        self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
        self.traffic_light_class = 'traffic light'
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.detection_active = False

        # Helmet detection is only valid when the loaded model actually
        # contains helmet-related classes. COCO yolov8n does not include
        # a helmet class, so "No Helmet" fines must be disabled in that case.
        self.helmet_detection_supported = self._has_helmet_class()
        self.allow_fallback_no_helmet = os.getenv('ALLOW_HELMET_FALLBACK', '0') == '1'
        if not self.helmet_detection_supported:
            if self.allow_fallback_no_helmet:
                print("⚠️  Helmet class not found in model. Using fallback no-helmet heuristic.")
            else:
                print("⚠️  Helmet class not found in model. No-helmet auto-fining disabled (set ALLOW_HELMET_FALLBACK=1 to enable heuristic).")

    def _has_helmet_class(self):
        """Return True if loaded model defines at least one helmet-like class."""
        names = getattr(self.model, 'names', {})
        if isinstance(names, dict):
            labels = [str(v).lower() for v in names.values()]
        elif isinstance(names, (list, tuple)):
            labels = [str(v).lower() for v in names]
        else:
            labels = []

        return any('helmet' in label for label in labels)
        
    def detect_objects(self, frame):
        """
        Detect objects in a frame
        Args:
            frame: OpenCV image frame
        Returns:
            results: Detection results
        """
        results = self.model(frame, conf=self.confidence_threshold)
        return results
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in frame
        Returns:
            List of detected vehicles with bounding boxes
        """
        results = self.detect_objects(frame)
        vehicles = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                if class_name in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    vehicles.append({
                        'type': class_name,
                        'confidence': confidence,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
        
        return vehicles
    
    def detect_helmet(self, frame, person_bbox=None):
        """
        Detect if motorcycle rider is wearing helmet
        Args:
            frame: Image frame
            person_bbox: Bounding box of person to check
        Returns:
            Boolean: True if helmet detected, False otherwise
        """
        # This is a simplified version.
        # If model has no helmet class, use a conservative fallback heuristic.
        
        if person_bbox:
            x1, y1, x2, y2 = person_bbox
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                return None

            if not self.helmet_detection_supported:
                if not self.allow_fallback_no_helmet:
                    return None
                return self._fallback_helmet_check(roi)

            # Use YOLO to detect objects in ROI
            results = self.model(roi, conf=0.3)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_name = result.names[int(box.cls[0])].lower()
                    # Only treat explicit helmet classes as helmets.
                    # Caps/hats are now considered NO helmet.
                    if 'helmet' in class_name:
                        return True
        
        return False

    def _fallback_helmet_check(self, roi):
        """Heuristic fallback when no helmet class exists in the loaded model.

        Returns:
            True  -> likely helmet present
            False -> likely no helmet
            None  -> uncertain (skip fining)
        """
        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return None

        # Focus on upper rider region where helmet/head appears.
        head = roi[0:max(int(h * 0.65), 1), :]
        hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)

        # Broad skin-tone masks in HSV (robust across lighting, still approximate).
        skin1 = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([25, 200, 255]))
        skin2 = cv2.inRange(hsv, np.array([160, 20, 40]), np.array([180, 200, 255]))
        skin_mask = cv2.bitwise_or(skin1, skin2)

        skin_ratio = cv2.countNonZero(skin_mask) / max(head.shape[0] * head.shape[1], 1)

        # Conservative thresholds to reduce false fines:
        # - clear visible skin on head area => likely no helmet
        # - very low visible skin => likely helmet/full-face coverage
        if skin_ratio >= 0.14:
            return False
        if skin_ratio <= 0.04:
            return True

        return None
    
    def detect_traffic_light(self, frame):
        """
        Detect traffic lights and their status
        Returns:
            List of traffic lights with status (red/green/yellow)
        """
        results = self.detect_objects(frame)
        traffic_lights = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_name = result.names[int(box.cls[0])]
                
                if 'traffic light' in class_name.lower():
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Extract traffic light region
                    tl_roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    
                    # Detect color (simplified - uses color detection)
                    status = self._detect_light_color(tl_roi)
                    
                    traffic_lights.append({
                        'status': status,
                        'bbox': bbox,
                        'confidence': float(box.conf[0])
                    })
        
        return traffic_lights
    
    def _detect_light_color(self, light_roi):
        """
        Detect traffic light color from ROI
        Returns: 'red', 'yellow', or 'green'
        """
        if light_roi.size == 0:
            return 'unknown'
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([40, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2
        
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        # Determine dominant color
        max_pixels = max(red_pixels, green_pixels, yellow_pixels)
        
        if max_pixels == red_pixels and red_pixels > 10:
            return 'red'
        elif max_pixels == green_pixels and green_pixels > 10:
            return 'green'
        elif max_pixels == yellow_pixels and yellow_pixels > 10:
            return 'yellow'
        else:
            return 'unknown'
    
    def draw_detections(self, frame, vehicles, traffic_lights):
        """
        Draw bounding boxes on frame
        """
        output_frame = frame.copy()
        
        # Draw vehicles
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{vehicle['type']}: {vehicle['confidence']:.2f}"
            cv2.putText(output_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw traffic lights
        for light in traffic_lights:
            x1, y1, x2, y2 = light['bbox']
            color = (0, 0, 255) if light['status'] == 'red' else (0, 255, 0)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_frame, light['status'].upper(), (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_frame

if __name__ == "__main__":
    # Test the detector
    print("Testing TrafficEye AI Detector...")
    detector = TrafficDetector()
    print("✅ Detector initialized successfully!")
