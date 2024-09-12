
import os
import requests
import zipfile
from tqdm import tqdm

def prepare_esc50_dataset(path="data/"):

    link = "https://github.com/karoldvl/ESC-50/archive/master.zip"

    if not os.path.exists(path + "ESC-50-master"):

        # Download the zip file
        print("Downloading ESC-50 dataset...")
        r = requests.get(link, stream=True)

        with open(path + "ESC-50-master.zip", 'wb') as file:
            for chunk in tqdm(r.iter_content(chunk_size=1024)):
                if chunk:
                    file.write(chunk)

        # Unzip the file
        with zipfile.ZipFile(path + "ESC-50-master.zip", 'r') as zip_ref:
            zip_ref.extractall(path)

        # Remove the zip file
        os.remove(path + "ESC-50-master.zip")
    else:
        print("ESC-50 dataset already exists")


if __name__ == '__main__':
    prepare_esc50_dataset()