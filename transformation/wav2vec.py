from .base import Transformator
from joblib import Memory
from utils.wav2vec import extract_features, load

memory = Memory('/tmp', verbose=0)

class Wav2VecTransformator(Transformator):
    def transform(self, dataset, labels):
        cached_extract = memory.cache(extract_features)
        return cached_extract(dataset), labels