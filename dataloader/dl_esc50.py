import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


class ESC50DataLoader(Dataset):
    def __init__(self, path, subset: str = None, transform=None):
        self.transform = transform

        if subset=='training':
            files = Path(path).glob('[1-4]-*')
        elif subset=='validation':
            files = Path(path).glob('5-*')
        elif subset=='testing':
            files = Path(path).glob('4-*')
        else:
            raise ValueError(f"Given subset {subset} is not supported. Please choose from 'training' or 'validation'")
        
        self.items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
        self.length = len(self.items)
        
    def __getitem__(self, index):
        filename, label = self.items[index]
        waveform, sample_rate = torchaudio.load(filename)

        waveform = self.normalize(waveform, 32768.0)
        
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, int(label)
    
    def __len__(self):
        return self.length
    
    def normalize(self, audio, factor):
        return audio / factor
    
    def get_class_weights(self):
        n_classes = len(set([label for _, label in self.items]))
        class_counts = [0] * n_classes

        # Store transform to disable it
        transform = self.transform
        self.transform = None

        for i in tqdm(range(len(self))):
            _, label = self[i]
            class_counts[label] += 1

        total = sum(class_counts)
        class_weights = [total / count for count in class_counts]

        # Restore transform
        self.transform = transform

        return class_weights