import os

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from tflite_support.task.vision import ObjectDetector


def get_detector(
    model_name: str, max_results: int = 3, score_threshold: float = 0.3
) -> ObjectDetector:
    base_options = core.BaseOptions(file_name=os.path.join("./models", model_name))
    detection_options = processor.DetectionOptions(
        max_results=max_results, score_threshold=score_threshold
    )
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options
    )
    return vision.ObjectDetector.create_from_options(options)
