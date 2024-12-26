from .base import Transformator
from joblib import Memory
from utils.plain_vec import extract_features

memory = Memory('/tmp', verbose=0)

class PlainTransformator(Transformator):
    def transform(self, dataset, labels):
        cached_extract = memory.cache(extract_features)
        return cached_extract(dataset), labels