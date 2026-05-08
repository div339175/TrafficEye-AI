"""
TrafficEye AI - e-Challan Generation System
Generates digital challans for traffic violations (Indian format)
Integrates with Parivahan-style challan format
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from io import BytesIO

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class EChallanGenerator:
    """Generate Indian-format e-Challans for traffic violations"""

    # Fine amounts as per Motor Vehicles (Amendment) Act, 2019 (India)
    FINE_SCHEDULE = {
        'Red Light Violation': {
            'fine': 1000,
            'section': 'Section 119/177',
            'description': 'Jumping Red Light Signal',
            'points': 4,
        },
        'No Helmet': {
            'fine': 1000,
            'section': 'Section 129/177A',
            'description': 'Riding without Helmet',
            'points': 3,
        },
        'Wrong Lane': {
            'fine': 500,
            'section': 'Section 177',
            'description': 'Driving in Wrong Lane',
            'points': 2,
        },
        'Overspeeding': {
            'fine': 2000,
            'section': 'Section 183',
            'description': 'Exceeding Speed Limit',
            'points': 4,
        },
        'Triple Riding': {
            'fine': 1000,
            'section': 'Section 128/177',
            'description': 'More than two persons on two-wheeler',
            'points': 3,
        },
        'No Seatbelt': {
            'fine': 1000,
            'section': 'Section 194B',
            'description': 'Driving without Seatbelt',
            'points': 2,
        },
        'Using Mobile Phone': {
            'fine': 5000,
            'section': 'Section 184',
            'description': 'Using mobile phone while driving',
            'points': 4,
        },
    }

    # Payment status
    STATUS_PENDING = 'Pending'
    STATUS_PAID = 'Paid'
    STATUS_DISPUTED = 'Disputed'
    STATUS_OVERDUE = 'Overdue'

    def __init__(self, output_dir='static/challans'):
        """Initialize e-Challan generator"""
        self.output_dir = output_dir
        self.challan_counter = 0
        os.makedirs(output_dir, exist_ok=True)
        print("✅ e-Challan Generator initialized")

    def generate_challan(self, violation_data):
        """
        Generate an e-Challan for a violation.

        Args:
            violation_data: dict with keys:
                - violation_type: str (must match FINE_SCHEDULE)
                - vehicle_type: str
                - plate_number: str or None
                - location: str
                - evidence_image: str (filename)
                - confidence: float
                - timestamp: datetime

        Returns:
            dict with full challan details
        """
        self.challan_counter += 1
        violation_type = violation_data.get('violation_type', 'Red Light Violation')

        # Get fine info from schedule
        fine_info = self.FINE_SCHEDULE.get(violation_type, {
            'fine': 500,
            'section': 'Section 177',
            'description': 'General Traffic Violation',
            'points': 1,
        })

        timestamp = violation_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                timestamp = datetime.now()

        # Generate unique challan number
        challan_number = self._generate_challan_number(timestamp)

        # Generate payment reference ID
        payment_ref = self._generate_payment_ref(challan_number)

        # Due date is 60 days from violation
        due_date = timestamp + timedelta(days=60)

        challan = {
            'challan_number': challan_number,
            'payment_ref': payment_ref,
            'violation_type': violation_type,
            'violation_section': fine_info['section'],
            'violation_description': fine_info['description'],
            'fine_amount': fine_info['fine'],
            'penalty_points': fine_info['points'],
            'vehicle_type': violation_data.get('vehicle_type', 'Unknown'),
            'plate_number': violation_data.get('plate_number', 'Not Detected'),
            'location': violation_data.get('location', 'Junction A'),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'due_date': due_date.strftime('%Y-%m-%d'),
            'evidence_image': violation_data.get('evidence_image', ''),
            'confidence': violation_data.get('confidence', 0.0),
            'status': self.STATUS_PENDING,
            'issuing_authority': 'TrafficEye AI - Automated System',
            'state': self._get_state_from_plate(
                violation_data.get('plate_number', '')
            ),
            'late_fee': fine_info['fine'] * 2,  # Double fine after due date
        }

        # Generate challan image
        challan_image = self._generate_challan_image(challan)
        if challan_image:
            challan['challan_image'] = challan_image

        return challan

    def _generate_challan_number(self, timestamp):
        """Generate unique challan number in Indian format"""
        date_part = timestamp.strftime('%Y%m%d')
        seq = str(self.challan_counter).zfill(6)
        return f"TE-{date_part}-{seq}"

    def _generate_payment_ref(self, challan_number):
        """Generate payment reference for online payment"""
        hash_val = hashlib.md5(challan_number.encode()).hexdigest()[:8].upper()
        return f"PAY-{hash_val}"

    def _get_state_from_plate(self, plate_number):
        """Extract state from Indian plate number"""
        if not plate_number or len(plate_number) < 2:
            return 'Unknown'

        state_codes = {
            'DL': 'Delhi', 'MH': 'Maharashtra', 'KA': 'Karnataka',
            'TN': 'Tamil Nadu', 'AP': 'Andhra Pradesh', 'TS': 'Telangana',
            'UP': 'Uttar Pradesh', 'RJ': 'Rajasthan', 'GJ': 'Gujarat',
            'WB': 'West Bengal', 'MP': 'Madhya Pradesh', 'KL': 'Kerala',
            'PB': 'Punjab', 'HR': 'Haryana', 'BR': 'Bihar',
            'CG': 'Chhattisgarh', 'JH': 'Jharkhand', 'UK': 'Uttarakhand',
            'HP': 'Himachal Pradesh', 'GA': 'Goa', 'AS': 'Assam',
            'CH': 'Chandigarh', 'JK': 'Jammu & Kashmir', 'OD': 'Odisha',
        }
        code = plate_number.replace(' ', '')[:2].upper()
        return state_codes.get(code, 'Unknown')

    def _generate_challan_image(self, challan):
        """Generate a visual challan receipt image using OpenCV"""
        # Create challan image (A4-ish aspect ratio)
        width, height = 800, 1000
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Header - Red banner
        cv2.rectangle(img, (0, 0), (width, 120), (0, 0, 180), -1)
        cv2.putText(img, "e-CHALLAN", (250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(img, "Traffic Violation Notice - TrafficEye AI",
                    (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

        # Challan details
        y_pos = 160
        line_height = 35

        details = [
            ("Challan No:", challan['challan_number']),
            ("Date/Time:", challan['timestamp']),
            ("", ""),
            ("VIOLATION DETAILS", ""),
            ("Type:", challan['violation_type']),
            ("Section:", f"{challan['violation_section']} MV Act"),
            ("Description:", challan['violation_description']),
            ("", ""),
            ("VEHICLE DETAILS", ""),
            ("Vehicle Type:", challan['vehicle_type']),
            ("Number Plate:", challan['plate_number']),
            ("State:", challan['state']),
            ("Location:", challan['location']),
            ("", ""),
            ("FINE DETAILS", ""),
            ("Fine Amount:", f"Rs. {challan['fine_amount']}/-"),
            ("Late Fee:", f"Rs. {challan['late_fee']}/- (after {challan['due_date']})"),
            ("Penalty Points:", str(challan['penalty_points'])),
            ("Due Date:", challan['due_date']),
            ("", ""),
            ("Payment Ref:", challan['payment_ref']),
            ("Status:", challan['status']),
            ("AI Confidence:", f"{challan['confidence']:.1%}"),
        ]

        for label, value in details:
            if label in ("VIOLATION DETAILS", "VEHICLE DETAILS",
                         "FINE DETAILS"):
                # Section header
                y_pos += 5
                cv2.rectangle(img, (30, y_pos - 18), (width - 30, y_pos + 8),
                              (240, 240, 240), -1)
                cv2.putText(img, label, (40, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
                y_pos += line_height
                continue

            if not label and not value:
                y_pos += 10
                continue

            cv2.putText(img, label, (40, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
            cv2.putText(img, str(value), (250, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            y_pos += line_height

        # Footer
        cv2.rectangle(img, (0, height - 80), (width, height), (240, 240, 240), -1)
        cv2.putText(img, "Pay online at: echallan.parivahan.gov.in",
                    (150, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 180), 2)
        cv2.putText(img, "This is a computer-generated challan. No signature required.",
                    (100, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (120, 120, 120), 1)

        # Border
        cv2.rectangle(img, (5, 5), (width - 5, height - 5), (0, 0, 180), 2)

        # Save
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"challan_{challan['challan_number']}_{timestamp_str}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, img)

        return filename

    def get_fine_amount(self, violation_type):
        """Get fine amount for a violation type"""
        info = self.FINE_SCHEDULE.get(violation_type, {})
        return info.get('fine', 500)

    def calculate_total_fines(self, challans):
        """Calculate total fines from list of challans"""
        total = sum(c.get('fine_amount', 0) for c in challans)
        return total

    def get_overdue_challans(self, challans):
        """Get list of overdue challans"""
        now = datetime.now()
        overdue = []
        for c in challans:
            due = datetime.strptime(c['due_date'], '%Y-%m-%d')
            if due < now and c['status'] == self.STATUS_PENDING:
                c['status'] = self.STATUS_OVERDUE
                overdue.append(c)
        return overdue


if __name__ == "__main__":
    print("Testing e-Challan Generator...")
    gen = EChallanGenerator()

    test_violation = {
        'violation_type': 'Red Light Violation',
        'vehicle_type': 'car',
        'plate_number': 'MH 12 AB 1234',
        'location': 'Junction A, Mumbai',
        'evidence_image': 'test.jpg',
        'confidence': 0.92,
        'timestamp': datetime.now(),
    }

    challan = gen.generate_challan(test_violation)
    print(f"✅ Challan generated: {challan['challan_number']}")
    print(f"   Fine: Rs. {challan['fine_amount']}")
    print(f"   Section: {challan['violation_section']}")
    print(f"   Payment Ref: {challan['payment_ref']}")
