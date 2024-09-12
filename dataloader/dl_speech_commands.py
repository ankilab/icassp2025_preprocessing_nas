from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio
import os
from pathlib import Path
from tqdm import tqdm

# Code from: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html#training-and-testing-the-network

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, path: str = "data/.", subset: str = None, transform=None):
        super().__init__(path, download=True)
        self.transform = transform

        self.classes = {
                    'yes': 0,
                    'no': 1,
                    'up': 2,
                    'down': 3,
                    'left': 4,
                    'right': 5,
                    'on': 6,
                    'off': 7,
                    'stop': 8,
                    'go': 9,
                    '_silence_': 10,
                    '_unknown_': 11
                }

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            self._walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def __getitem__(self, n):
        # file path
        fileid = self._walker[n]
        # load wavefile
        waveform, sample_rate = torchaudio.load(fileid)
        
        label = self._walker[n].split('/')[3]
        if label not in self.classes:
            label = '_unknown_'
        label = self.classes[label]

        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label
    
    def get_class_weights(self):
        class_counts = [0] * len(self.classes)

        # Store transform to disable it
        transform = self.transform
        self.transform = None

        for i in tqdm(range(len(self))):
            _, label = self[i]
            class_counts[label] += 1
        total_samples = sum(class_counts)
        class_weights = [total_samples / count for count in class_counts]

        # Restore transform
        self.transform = transform

        return class_weights

