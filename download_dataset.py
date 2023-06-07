import requests
import zipfile
import os


def _download_extract(url, data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    zip_filename = os.path.join(data_dir, url.split("/")[-1])
    print(f"Downloading {zip_filename}...")
    response = requests.get(url)
    with open(zip_filename, "wb") as f:
        f.write(response.content)
    print(f"Extracting {zip_filename}")
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(zip_filename)
    print("Done")


def download():
    dataset_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
    data_dir = "./data"

    _download_extract(dataset_url, data_dir)
    _download_extract(annotations_url, data_dir)
