"""
TrafficEye AI - Run Detection (Enhanced)
Now includes: ANPR, e-Challan, Speed Estimation, Severity Scoring,
              Vehicle Color, Day/Night Mode, Triple Riding Detection
"""

import cv2
import sys
from detection.detector import TrafficDetector
from detection.violation_logic import ViolationDetector
from detection.anpr import NumberPlateRecognizer
from detection.echallan import EChallanGenerator
from detection.innovations import (
    SpeedEstimator, SeverityScorer, DayNightDetector,
    VehicleColorDetector, TripleRidingDetector,
    PeakHourAnalyzer, ViolationHeatmap
)
from app import db, Violation, Challan, app
from datetime import datetime


def run_detection(video_source=0, display=True):
    """
    Run traffic violation detection with all features
    """
    print("=" * 60)
    print("🚦 TrafficEye AI - Starting Enhanced Detection")
    print("=" * 60)

    # Initialize all modules
    traffic_detector = TrafficDetector()
    violation_detector = ViolationDetector()
    anpr = NumberPlateRecognizer()
    challan_gen = EChallanGenerator()
    speed_estimator = SpeedEstimator()
    severity_scorer = SeverityScorer()
    day_night = DayNightDetector()
    color_detector = VehicleColorDetector()
    triple_detector = TripleRidingDetector()
    peak_analyzer = PeakHourAnalyzer()
    heatmap = ViolationHeatmap()

    print("✅ All modules initialized")

    # Open video source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {video_source}")
        return

    # Get frame dimensions and set stop line
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        stop_line_y = int(height * 0.6)
        violation_detector.set_stop_line(stop_line_y)
        heatmap = ViolationHeatmap(width, height)
        print(f"📏 Frame: {width}x{height}, Stop line at Y={stop_line_y}")

    print("✅ Detection started. Press 'q' to quit.")
    print("=" * 60)

    frame_count = 0
    process_every_n_frames = 3

    with app.app_context():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  End of video or cannot read frame")
                break

            frame_count += 1

            if frame_count % process_every_n_frames == 0:
                # Day/Night detection & frame enhancement
                lighting = day_night.detect_mode(frame)
                enhanced_frame = day_night.enhance_night_frame(frame)

                # Detect vehicles
                vehicles = traffic_detector.detect_vehicles(enhanced_frame)

                # Estimate speed for each vehicle
                vehicles = speed_estimator.update(vehicles, frame_count)

                # Detect traffic lights
                traffic_lights = traffic_detector.detect_traffic_light(enhanced_frame)

                # Check for violations
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

                # Process each violation: ANPR + Color + Severity + e-Challan
                for violation in violations:
                    # ANPR - Read number plate
                    plate_result = anpr.recognize_plate(
                        enhanced_frame, violation.get('bbox')
                    )
                    plate_number = ''
                    if plate_result:
                        plate_number = plate_result['text']
                        print(f"🔢 Plate detected: {plate_number}")

                    # Vehicle color
                    color_result = color_detector.detect_color(
                        enhanced_frame, violation['bbox']
                    )

                    # Severity scoring
                    severity = severity_scorer.calculate_severity(violation)

                    # Peak hour logging
                    peak_analyzer.log_violation(violation['type'])

                    # Heatmap
                    cx = (violation['bbox'][0] + violation['bbox'][2]) // 2
                    cy = (violation['bbox'][1] + violation['bbox'][3]) // 2
                    heatmap.add_violation(cx, cy)

                    # Save to database
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

                    print(f"🚨 {violation['type']} | Plate: {plate_number or 'N/A'} | "
                          f"Severity: {severity['level']} ({severity['score']}) | "
                          f"Challan: {challan_data['challan_number']} | "
                          f"Fine: ₹{challan_data['fine_amount']}")

                # Draw everything on display frame
                display_frame = traffic_detector.draw_detections(
                    frame, vehicles, traffic_lights
                )
                display_frame = violation_detector.draw_stop_line(display_frame)
                display_frame = speed_estimator.draw_speed(display_frame, vehicles)

                # Stats overlay
                cv2.putText(display_frame, f"Vehicles: {len(vehicles)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Mode: {lighting.upper()}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Violations: {len(violation_detector.violations_detected)}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                display_frame = frame

            if display:
                cv2.imshow('TrafficEye AI - Enhanced Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Save heatmap
    heatmap.save_heatmap()

    cap.release()
    cv2.destroyAllWindows()

    # Print summary
    peak_hours = peak_analyzer.get_peak_hours()
    print("\n" + "=" * 60)
    print("✅ Detection completed!")
    print(f"📊 Total violations: {len(violation_detector.violations_detected)}")
    if peak_hours:
        print(f"⏰ Peak violation hour: {peak_hours[0]['label']} ({peak_hours[0]['count']} violations)")
    print("=" * 60)


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    print(f"📹 Using video source: {source}")
    run_detection(source)
