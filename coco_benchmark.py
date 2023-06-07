import argparse
import os
import json

import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from tqdm import tqdm

from detect import detect
from detector import get_detector

DATA_DIR = "./data"
DATA_TYPE = "val2017"


def select_images(coco, sample_size):
    img_ids = coco.getImgIds()
    selected_img_ids = np.random.choice(img_ids, sample_size)
    return coco.loadImgs(selected_img_ids)


def get_stats(coco):
    coco_gt = coco
    coco_dt = coco_gt.loadRes("stats/detections.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def eval_model(coco, detector, images, draw_boxes):
    all_detections = []

    for img_info in tqdm(images, desc="Evaluating model on COCO dataset"):
        image_path = os.path.join(DATA_DIR, DATA_TYPE, img_info["file_name"])
        detection_result, inference_time = detect(detector, image_path, draw_boxes)
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            all_detections.append(
                {
                    "image_id": img_info["id"],
                    "category_id": coco.getCatIds(
                        catNms=[detection.categories[0].category_name]
                    )[0],
                    "bbox": [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height],
                    "score": detection.categories[0].score,
                }
            )
    return all_detections, inference_time


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
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to use for detection. Defaults to 10",
    )

    parser.add_argument(
        "--draw_bbox",
        action="store_true",
        help="Boolean switch to decide whether to draw bounding boxes or not",
    )

    args = parser.parse_args()

    model_name = args.model_name
    max_results = args.max_results
    score_threshold = args.score_threshold
    num_samples = args.num_samples
    draw_bbox = args.draw_bbox

    ann_file = f"{DATA_DIR}/annotations/instances_{DATA_TYPE}.json"
    coco = COCO(ann_file)
    detector = get_detector(model_name, max_results, score_threshold)
    images = select_images(coco, num_samples)

    all_detections, inference_time = eval_model(
        coco, detector, images, draw_boxes=draw_bbox
    )

    with open("stats/detections.json", "w") as f:
        json.dump(all_detections, f)
    get_stats(coco)
    print(
        f"Mean inference time for per image: {round(np.array(inference_time).mean(), 4)} sec"
    )


if __name__ == "__main__":
    main()
