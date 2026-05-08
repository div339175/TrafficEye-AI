# Detection module for TrafficEye AI
from detection.detector import TrafficDetector
from detection.violation_logic import ViolationDetector
from detection.anpr import NumberPlateRecognizer
from detection.echallan import EChallanGenerator
from detection.innovations import (
    SpeedEstimator, ViolationHeatmap, PeakHourAnalyzer,
    SeverityScorer, DayNightDetector, VehicleColorDetector,
    TripleRidingDetector, PredictiveAnalytics
)

__all__ = [
    'TrafficDetector', 'ViolationDetector', 'NumberPlateRecognizer',
    'EChallanGenerator', 'SpeedEstimator', 'ViolationHeatmap',
    'PeakHourAnalyzer', 'SeverityScorer', 'DayNightDetector',
    'VehicleColorDetector', 'TripleRidingDetector', 'PredictiveAnalytics',
]
