from pathlib import Path
import shutil
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Set up paths
root_path = Path(__file__).parent.parent / 'dataset'
raw_path = root_path / 'raw'
ori_path = root_path / 'expanded'

# Define mappings
lang_mapping = {
    'Chinese': 'ZH',
    'English': 'EN',
}
type_mapping = {
    'True Story': 'T',
    'Deceptive Story': 'F'
}

# Read CSV attributes
attributes = pd.read_csv(raw_path / 'story_attributes.csv')

# Clear and create the expanded directory
shutil.rmtree(ori_path, ignore_errors=True)
ori_path.mkdir(parents=True, exist_ok=True)

# Data augmentation functions
def time_stretch(y, rate=1.2):
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift(y, sr, n_steps=4):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def change_volume(y, factor=1.2):
    return y * factor

def frequency_masking(y, frequency_mask_param=10):
    S = librosa.stft(y)
    mask = np.ones_like(S)
    num_freqs = S.shape[0]
    freq_mask_start = np.random.randint(0, num_freqs - frequency_mask_param)
    mask[freq_mask_start:freq_mask_start + frequency_mask_param, :] = 0
    return librosa.istft(S * mask)

# Process and augment audio
def process_audio(file):
    y, sr = librosa.load(file, sr=16000)

    augmented_audios = []
    augmented_audios.append(time_stretch(y))  # Time stretch
    augmented_audios.append(pitch_shift(y, sr))  # Pitch shift
    augmented_audios.append(add_noise(y))  # Add noise
    augmented_audios.append(change_volume(y))  # Change volume
    augmented_audios.append(frequency_masking(y))  # Frequency masking

    data = [y] + augmented_audios
    for i in range(len(data)):
        data[i] = librosa.util.fix_length(data[i], size=sr*300)
    return data


def expand_wav():
    cnt = 1
    new_file_format = '{num}_{lang}_{type}_{augmented}.wav'

    for _, row in tqdm(attributes.iterrows(), total=attributes.shape[0], desc="Processing audio files"):
        file = Path(raw_path / row['filename'])
        file_type = type_mapping[row['Story_type']]
        lang = lang_mapping[row['Language']]

        if not file.exists():
            continue

        # Process the original file and its augmentations
        augmented_audios = process_audio(file)

        for i, audio in enumerate(augmented_audios):
            # Construct a new filename
            augmented_file = new_file_format.format(
                num=file.stem,
                lang=lang,
                type=file_type,
                augmented=f"aug{i + 1}"
            )
            augmented_file_path = ori_path / augmented_file

            # Save the augmented file
            sf.write(augmented_file_path, audio, 16000)
            cnt += 1

if __name__ == '__main__':
    expand_wav()
