import argparse
import os
import time

import cv2
from tflite_support.task import vision

from detector import get_detector


def put_fps(frame, exec_time):
    fps = 1 / exec_time
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )


def draw_bbox(image, detection):
    bbox = detection.bounding_box
    # Draw bounding box
    cv2.rectangle(
        image,
        (bbox.origin_x, bbox.origin_y),
        (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
        (0, 255, 0),
        2,
    )

    # Draw label
    cv2.putText(
        image,
        f"{detection.categories[0].category_name}: {detection.categories[0].score}",
        (bbox.origin_x, bbox.origin_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def run_detection(image, detector):
    start_time = time.time()
    image = preprocess_image(image)
    input_tensor = vision.TensorImage.create_from_array(image)
    detection_result = detector.detect(input_tensor)
    inference_time = time.time() - start_time
    return detection_result, inference_time


def detect(detector, source, draw_boxes=True):
    if source == "0" or source.endswith(".mov"):
        source = 0 if source == "0" else source
        vid = cv2.VideoCapture(source)
        while vid.isOpened():
            ret, frame = vid.read()
            detection_result, inference_time = run_detection(frame, detector)
            for detection in detection_result.detections:
                draw_bbox(frame, detection)
            put_fps(frame, inference_time)

            cv2.imshow("detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        vid.release()
        cv2.destroyAllWindows()
    else:
        image = cv2.imread(source)
        detection_result, inference_time = run_detection(image, detector)
        if draw_boxes:
            for detection in detection_result.detections:
                draw_bbox(image, detection)
            cv2.imshow("detector", image)
            cv2.waitKey(0)
    return detection_result, inference_time


# python detect.py --model_name=test.tflite --max_results=3 --score_threshold=0.3 --source=[image.jpg|vid.mp3|0(webcamera)]
def main():
    parser = argparse.ArgumentParser(description="Object detection script")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name for the detection in tflite format",
    )

    parser.add_argument(
        "--max_results",
        type=int,
        default=5,
        help="Maximum results to be returned after detection",
    )

    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Threshold for the detection score",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source for the object detection. It could be an image file, a video file, or '0' for the web camera",
    )

    args = parser.parse_args()

    model_name = args.model_name
    max_results = args.max_results
    score_threshold = args.score_threshold
    source = args.source

    print(
        f"Model Name: {model_name}, Max Results: {max_results}, Score Threshold: {score_threshold}, Source: {source}"
    )
    print(os.getcwd())

    if not os.path.exists(os.path.join("./models", model_name)):
        raise FileNotFoundError(model_name)

    if source != "0" and not os.path.exists(source):
        raise FileNotFoundError(source)

    detector = get_detector(model_name, max_results, score_threshold)
    detect(detector, source)


if __name__ == "__main__":
    main()
