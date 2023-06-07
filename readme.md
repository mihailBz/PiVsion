# Object detection using EfficientDet Lite

This project is designed to detect objects using EfficientDet lite or other models using Tensorflow Lite framework.

## Getting Started

### Prerequisites

Make sure you have Python 3.8.

### Setup

1. Set up a virtual environment (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate
    ```

2. Install the necessary packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Setup the project:

    ```bash
    python setup.py
    ```

## Running the Detection

To run detection on a user input, use the `detect.py` script:

```bash
python detect.py --model_name=efficientdet_lite0.tflite --max_results=3 --score_threshold=0.3 --source=image.png
```
It can be an image, video or --source=0 for web camera.

## Benchmarking on the COCO Dataset

To run detection and collect performance statistics on the COCO dataset, use the coco_benchmark.py script:

```bash
python coco_benchmark.py --model_name=efficientdet_lite0.tflite --max_results=3 --score_threshold=0.3 --num_samples=10 --draw_bbox
```

The num_samples parameter can be adjusted based on the number of samples to be evaluated. The draw_bbox switch can be used to decide whether to draw bounding boxes in the output images.