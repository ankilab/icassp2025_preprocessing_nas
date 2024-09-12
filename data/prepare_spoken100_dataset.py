import os
import requests
import zipfile
from tqdm import tqdm

def prepare_spoken100_dataset(path="data/"):

    link = "https://zenodo.org/records/10810044/files/SpokeN-100.zip?download=1"

    if not os.path.exists(path + "SpokeN-100"):
        # Download the zip file
        print("Downloading SpokeN-100 dataset...")
        r = requests.get(link, stream=True)

        with open(path + "SpokeN-100.zip", 'wb') as file:
            for chunk in tqdm(r.iter_content(chunk_size=1024)):
                if chunk:
                    file.write(chunk)

        # Unzip the file
        with zipfile.ZipFile(path + "SpokeN-100.zip", 'r') as zip_ref:
            zip_ref.extractall(path)

        # Remove the zip file
        os.remove(path + "SpokeN-100.zip")
    else:
        print("SpokeN-100 dataset already exists")


if __name__ == '__main__':
    prepare_spoken100_dataset()