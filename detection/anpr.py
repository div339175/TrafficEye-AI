"""
TrafficEye AI - Automatic Number Plate Recognition (ANPR)
Detects and reads vehicle number plates using OpenCV + EasyOCR
"""

import cv2
import numpy as np
import os
import re
from datetime import datetime

# Try to import easyocr, fall back to basic OCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR not available. Using basic plate detection only.")


class NumberPlateRecognizer:
    """Automatic Number Plate Recognition using OpenCV + EasyOCR"""

    # Indian number plate regex patterns
    # Standard: XX 00 XX 0000 (e.g., MH 12 AB 1234)
    INDIAN_PLATE_PATTERNS = [
        r'[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4}',   # MH 12 AB 1234
        r'[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}',             # MH12AB1234
        r'[A-Z]{2}\s?\d{2}\s?[A-Z]{1,2}\s?\d{4}',        # MH 01 AZ 4567
    ]

    # Indian state RTO codes for validation
    INDIAN_STATE_CODES = [
        'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'GA',
        'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH',
        'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK',
        'TN', 'TR', 'TS', 'UK', 'UP', 'WB',
    ]

    def __init__(self):
        """Initialize the ANPR system"""
        print("🔧 Initializing Number Plate Recognition...")

        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("✅ EasyOCR initialized for plate reading")
            except Exception as e:
                print(f"⚠️  EasyOCR init failed: {e}. Using basic detection.")

        # Plate detection parameters
        self.min_plate_area = 500
        self.max_plate_area = 50000
        self.plate_aspect_ratio_range = (2.0, 6.0)  # Width/Height ratio

        # Cache to avoid duplicate reads
        self._recent_plates = {}
        self._cache_timeout = 10  # seconds

        print("✅ ANPR module ready")

    def detect_plate_region(self, frame, vehicle_bbox=None):
        """
        Detect number plate region in frame or within a vehicle bounding box.
        Uses edge detection + contour analysis.

        Args:
            frame: Full video frame
            vehicle_bbox: Optional (x1, y1, x2, y2) to search only within vehicle

        Returns:
            List of plate regions [(x1, y1, x2, y2), ...]
        """
        if vehicle_bbox:
            x1, y1, x2, y2 = vehicle_bbox
            # Focus on lower 50% of vehicle (where plates usually are)
            plate_region_y = y1 + int((y2 - y1) * 0.5)
            roi = frame[plate_region_y:y2, x1:x2]
            offset_x, offset_y = x1, plate_region_y
        else:
            roi = frame
            offset_x, offset_y = 0, 0

        if roi.size == 0:
            return []

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edges = cv2.Canny(filtered, 30, 200)

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plate_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_plate_area or area > self.max_plate_area:
                continue

            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Plates are roughly rectangular (4 corners)
            if len(approx) >= 4 and len(approx) <= 8:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / max(h, 1)

                if self.plate_aspect_ratio_range[0] <= aspect_ratio <= self.plate_aspect_ratio_range[1]:
                    # Convert back to full frame coordinates
                    plate_x1 = x + offset_x
                    plate_y1 = y + offset_y
                    plate_x2 = plate_x1 + w
                    plate_y2 = plate_y1 + h
                    plate_regions.append((plate_x1, plate_y1, plate_x2, plate_y2))

        # Sort by area (largest first) and return top candidates
        plate_regions.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
        return plate_regions[:3]

    def read_plate_text(self, frame, plate_bbox):
        """
        Read text from a detected plate region using OCR.

        Args:
            frame: Full video frame
            plate_bbox: (x1, y1, x2, y2) of plate region

        Returns:
            dict with 'text', 'confidence', 'is_indian', 'state_code'
        """
        x1, y1, x2, y2 = plate_bbox
        plate_img = frame[y1:y2, x1:x2]

        if plate_img.size == 0:
            return None

        # Preprocess plate image for better OCR
        plate_processed = self._preprocess_plate(plate_img)

        plate_text = ""
        ocr_confidence = 0.0

        if self.reader:
            try:
                results = self.reader.readtext(plate_processed, detail=1)
                if results:
                    # Combine all detected text segments
                    texts = []
                    confidences = []
                    for (bbox, text, conf) in results:
                        texts.append(text.upper().strip())
                        confidences.append(conf)
                    plate_text = ' '.join(texts)
                    ocr_confidence = np.mean(confidences) if confidences else 0.0
            except Exception:
                pass

        if not plate_text:
            return None

        # Clean up the text
        plate_text = self._clean_plate_text(plate_text)

        if not plate_text or len(plate_text) < 4:
            return None

        # Validate against Indian plate patterns
        is_indian, state_code = self._validate_indian_plate(plate_text)

        return {
            'text': plate_text,
            'confidence': float(ocr_confidence),
            'is_indian': is_indian,
            'state_code': state_code,
            'bbox': plate_bbox
        }

    def recognize_plate(self, frame, vehicle_bbox=None):
        """
        Full pipeline: detect plate region → read text.

        Args:
            frame: Video frame
            vehicle_bbox: Optional vehicle bounding box to search within

        Returns:
            Best plate result dict or None
        """
        plate_regions = self.detect_plate_region(frame, vehicle_bbox)

        best_result = None
        best_confidence = 0.0

        for plate_bbox in plate_regions:
            result = self.read_plate_text(frame, plate_bbox)
            if result and result['confidence'] > best_confidence:
                best_result = result
                best_confidence = result['confidence']

        # Check cache to avoid duplicates within timeout
        if best_result:
            plate_key = best_result['text']
            now = datetime.now().timestamp()
            if plate_key in self._recent_plates:
                if now - self._recent_plates[plate_key] < self._cache_timeout:
                    return None  # Already detected recently
            self._recent_plates[plate_key] = now

            # Clean old cache entries
            self._recent_plates = {
                k: v for k, v in self._recent_plates.items()
                if now - v < self._cache_timeout * 3
            }

        return best_result

    def _preprocess_plate(self, plate_img):
        """Preprocess plate image for better OCR accuracy"""
        # Resize to standard width
        target_width = 300
        h, w = plate_img.shape[:2]
        if w > 0:
            scale = target_width / w
            plate_img = cv2.resize(plate_img, (target_width, int(h * scale)))

        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.medianBlur(thresh, 3)

        return denoised

    def _clean_plate_text(self, text):
        """Clean OCR output to valid plate characters"""
        # Keep only alphanumeric and spaces
        cleaned = re.sub(r'[^A-Z0-9\s]', '', text.upper())
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        # Common OCR corrections for Indian plates
        replacements = {
            'O': '0', 'I': '1', 'S': '5', 'B': '8',
            'G': '6', 'Z': '2', 'Q': '0',
        }
        # Only apply to digit positions (basic heuristic)
        result = list(cleaned)
        for i, char in enumerate(result):
            if i > 2 and char in replacements and cleaned[:2].isalpha():
                # Likely a digit position after state code
                pass  # Keep as-is; context-dependent
        return cleaned

    def _validate_indian_plate(self, plate_text):
        """
        Validate if plate matches Indian number plate format.

        Returns:
            (is_indian: bool, state_code: str or None)
        """
        clean = plate_text.replace(' ', '')

        for pattern in self.INDIAN_PLATE_PATTERNS:
            if re.search(pattern, clean):
                # Extract state code (first 2 letters)
                state_code = clean[:2]
                if state_code in self.INDIAN_STATE_CODES:
                    return True, state_code
                return True, None

        return False, None

    def draw_plate_detection(self, frame, plate_result):
        """Draw plate detection on frame for visualization"""
        if not plate_result:
            return frame

        output = frame.copy()
        x1, y1, x2, y2 = plate_result['bbox']

        # Draw plate bounding box (yellow)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw plate text above the box
        text = plate_result['text']
        conf = plate_result['confidence']
        label = f"PLATE: {text} ({conf:.0%})"

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 255), -1)
        cv2.putText(output, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Indian plate indicator
        if plate_result['is_indian']:
            state = plate_result['state_code'] or '??'
            cv2.putText(output, f"🇮🇳 {state}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return output

    def save_plate_image(self, frame, plate_result, output_dir='static/evidence'):
        """Save cropped plate image as evidence"""
        if not plate_result:
            return None

        os.makedirs(output_dir, exist_ok=True)

        x1, y1, x2, y2 = plate_result['bbox']
        plate_crop = frame[y1:y2, x1:x2]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plate_text_clean = plate_result['text'].replace(' ', '_')
        filename = f"plate_{plate_text_clean}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, plate_crop)
        return filename


if __name__ == "__main__":
    print("Testing ANPR Module...")
    anpr = NumberPlateRecognizer()
    print("✅ ANPR module initialized successfully!")
