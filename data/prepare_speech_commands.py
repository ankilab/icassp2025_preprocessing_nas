from torchaudio.datasets import SPEECHCOMMANDS
import os
import random
import torchaudio
from pathlib import Path

def prepare_speech_commands():
    SPEECHCOMMANDS(root='data/', download=True)

    base_path = Path('data/SpeechCommands/speech_commands_v0.02/')
    background_noise_path = base_path / '_background_noise_'
    silence_folder = base_path / '_silence_'

    silence_files = []

    # create _silence_ folder
    os.makedirs(silence_folder, exist_ok=True)

    # go through all files in the background noise folder and split them into 1 second files into _silence_ folder
    for file in background_noise_path.glob('*.wav'):
        waveform, sample_rate = torchaudio.load(file)
        samples_per_second = sample_rate
        samples_per_file = samples_per_second
        for i in range(0, waveform.size(1) // samples_per_file):
            fname = silence_folder / (file.stem + str(i) + '.wav')
            torchaudio.save(fname, waveform[:, i*samples_per_file:(i+1)*samples_per_file], sample_rate)
            silence_files.append(os.path.join('_silence_', fname.name))
        if waveform.size(1) % samples_per_file != 0:
            fname = silence_folder / (file.stem + str(i+1) + '.wav')
            torchaudio.save(fname, waveform[:, (i+1)*samples_per_file:], sample_rate)
            silence_files.append(os.path.join('_silence_', fname.name))

    # get amount of samples for each class in testing and validation set
    num_test_samples_each_class = 300

    num_val_samples_each_class = 50

    # randomly split the _silence_ files into testing and validation set
    random.seed(0)
    random.shuffle(silence_files)
    silence_files_test = silence_files[:num_test_samples_each_class]
    silence_files_val = silence_files[num_test_samples_each_class:num_test_samples_each_class+num_val_samples_each_class]

    # write the files to the testing and validation list
    with open(base_path / 'testing_list.txt', 'a') as fileobj:
        for file in silence_files_test:
            fileobj.write(file + '\n')

    with open(base_path / 'validation_list.txt', 'a') as fileobj:
        for file in silence_files_val:
            fileobj.write(file + '\n')

    # Delete data/speech_commands_v0.02.tar.gz
    os.remove('data/SpeechCommands/speech_commands_v0.02.tar.gz')

    print("Done")


if __name__ == '__main__':
    prepare_speech_commands()
