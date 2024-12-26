from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import soundfile
from joblib import Memory

dataset_path = Path(__file__).parent.parent / 'dataset' / 'expanded'

@dataclass
class Audio:
    audio: np.ndarray | torch.Tensor
    sr: int
    language: np.ushort
    is_true: bool

lang_mapping = {
    'ZH': 0,
    'EN': 1,
}


memory = Memory('/tmp', verbose=0)

def load_data(backend='librosa', only_labels=False) -> tuple[list[Audio], list[bool]]:
    """
    Load the data from the dataset folder
    :return: List of tuples containing the librosa-loaded audio, sample rate, one-hot encoded language, and if it is a true story
    """
    dat = []
    label = []
    for path in dataset_path.iterdir():
        if only_labels:
            label.append(path.stem.split('_')[2] == 'T')
            continue
        if backend == 'librosa':
            audio, sr = librosa.load(path, sr=None)
        elif backend == 'torchaudio':
            audio, sr = torchaudio.load(path)
        elif backend == 'soundfile':
            audio, sr = soundfile.read(path)
        else:
            raise ValueError(f'Invalid backend: {backend}')
        idx, language, story_type, _ = path.stem.split('_', 3)
        language = np.ushort(lang_mapping[language])
        story_type = story_type == 'T'
        dat.append(Audio(audio, sr, language, story_type))
        label.append(story_type)
    return dat, label

load_data = memory.cache(load_data)

if __name__ == '__main__':
    print(load_data())