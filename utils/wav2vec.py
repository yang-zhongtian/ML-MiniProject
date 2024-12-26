from pathlib import Path
import numpy as np
from multiprocessing import Process, set_start_method
import tempfile

feature_file = Path(__file__).parent.parent / 'feature' / 'wav2vec.npz'


def __process_batch(batch, batch_idx, save_file):
    import torch, torchaudio
    import numpy as np
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
    model = bundle.get_model().to(device)
    model.eval()
    
    features = []
    
    for value in tqdm(batch, desc=f"Batch {batch_idx}"):
        y, sr = value.audio, value.sr

        waveform = y.to(device)

        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        with torch.inference_mode():
            feats, _ = model.extract_features(waveform)
            feature_vector = feats[-1][0]
        
        feature_vector = feature_vector.cpu().numpy()
        features.append(feature_vector)
    
    np.save(save_file, np.array(features))


def load():
    data = np.load(feature_file)
    features = data['features']
    labels = data['labels']
    return features, labels


def extract():
    from attributes import load_data
    print("Loading data")
    dataset, labels = load_data(backend='torchaudio')
    print("Data loaded")
    
    features = extract_features(dataset)
    
    np.savez(feature_file, features=features, labels=labels)
    print(f"Features saved to {feature_file}")



def extract_features(dataset):
    try:
        set_start_method('spawn')
    except RuntimeError:
        # The context has already been set, possibly by the environment
        pass

    feature_chunks = []

    with tempfile.TemporaryDirectory() as temp:
        save_dir = Path(temp)
        
        batch_size = 50
        
        for batch_idx in range(0, len(dataset) // batch_size):
            batch = dataset[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            save_file = save_dir / f"features_batch_{batch_idx}.npy"
            
            p = Process(target=__process_batch, args=(batch, batch_idx, save_file))
            p.start()
            p.join()

            batch_features = np.load(save_file)
            save_file.unlink()
            feature_chunks.append(batch_features)

    features = np.concatenate(feature_chunks, axis=0)
    return features

if __name__ == "__main__":
    extract()
