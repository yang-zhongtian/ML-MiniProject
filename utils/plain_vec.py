from pathlib import Path
import numpy as np
import librosa
from multiprocessing import Pool, cpu_count

feature_file = Path(__file__).parent.parent / 'feature' / 'plain.npz'

def load():
    data = np.load(feature_file)
    features = data['features']
    labels = data['labels']
    return features, labels

def __extract_features(y: np.array, sr: int, n_mfcc=13):
    """Extracts a comprehensive set of audio features from an audio signal."""
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Extract Chroma feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Extract Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)

    # Extract Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Extract Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)

    # Combine all features into a single feature vector
    feature_vector = np.hstack((
        mfccs_mean,
        chroma_mean,
        mel_mean,
        zcr_mean,
        spectral_centroid_mean
    ))

    return feature_vector

def process_single_data(data):
    """Helper function to extract features for a single data point."""
    y, sr, language = data.audio, data.sr, data.language
    feature_vector = __extract_features(y, sr)
    feature = np.hstack((feature_vector, language))
    return feature

def extract_features(dataset):
    from tqdm import tqdm

    num_processes = cpu_count()

    features = []
    with Pool(processes=num_processes) as pool:
        for feature in tqdm(pool.imap(process_single_data, dataset), total=len(dataset), desc='Extracting features'):
            features.append(feature)

    return np.array(features)

def extract():
    from attributes import load_data
    print("Loading data")
    dataset, labels = load_data()
    print("Data loaded")
    
    features = extract_features(dataset)
    
    np.savez(feature_file, features=features, labels=labels)
    print(f"Features saved to {feature_file}")

if __name__ == '__main__':
    extract()
