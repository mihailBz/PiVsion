import requests
import os
from tqdm import tqdm


def _download(url, models_dir="./models"):
    os.makedirs(models_dir, exist_ok=True)
    model_name = url.split("/")[-5] + "_" + url.split("/")[-4]
    model_filename = os.path.join(models_dir, model_name + ".tflite")
    response = requests.get(url)
    with open(model_filename, "wb") as f:
        f.write(response.content)


def download():
    models_urls = [
        "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite",
        "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1?lite-format=tflite",
        "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1?lite-format=tflite",
        "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite3/detection/metadata/1?lite-format=tflite",
        "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite4/detection/metadata/2?lite-format=tflite",
    ]
    print(f"Downloading pretrained models...")
    for url in tqdm(models_urls):
        _download(url)
    print("Done")
