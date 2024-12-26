from sklearn.linear_model import LogisticRegression
from model.base import Model
from transformation.base import Transformator
from pathlib import Path
import numpy as np
import joblib

default_base_dir = Path(__file__).parent.parent / 'weights'

class Ensembler:
    active_models: list[tuple[Transformator, Model]] = []
    meta_model: LogisticRegression
    base_dir: Path

    def __init__(self, meta_model: LogisticRegression, models: list[tuple[Transformator, Model]], base_dir=default_base_dir):
        self.meta_model = meta_model
        self.base_dir = base_dir
        self.active_models = models
    
    @staticmethod
    def create(models: list[tuple[Transformator, Model]], base_dir=default_base_dir) -> 'Ensembler':
        meta_model = LogisticRegression(max_iter=1000)
        return Ensembler(meta_model, models, base_dir)
    
    def fit(self, audios, labels):
        X_meta_train = []

        for _, transtormator, model in self.active_models:
            tf = transtormator()
            X, _ = tf.transform(audios, labels)
            x_meta = model.predict_proba(X)
            X_meta_train.append(x_meta)
            print(x_meta.shape)
        
        X_meta_train = np.hstack(X_meta_train)
        print(X_meta_train.shape)
        self.meta_model.fit(X_meta_train, labels)

    @staticmethod
    def load(meta_model_path: Path, models: list[tuple[Transformator, Model]]) -> 'Ensembler':
        meta_model = joblib.load(meta_model_path)
        return Ensembler(meta_model, models)
    
    def save(self, path: Path):
        joblib.dump(self.meta_model, path)
    
    def predict(self, audios):
        X_meta_test = []
        for transtormator, model in self.active_models:
            tf = transtormator()
            X, _ = tf.transform(audios, None)
            x_meta = model.predict_proba(X)
            print(x_meta.shape)
            X_meta_test.append(x_meta)
        
        X_meta_test = np.hstack(X_meta_test)
        print(X_meta_test.shape)
        return self.meta_model.predict(X_meta_test)
    
    def predict_proba(self, audios):
        X_meta_test = []
        for transtormator, model in self.active_models:
            tf = transtormator()
            X, _ = tf.transform(audios, None)
            x_meta = model.predict_proba(X)
            print(x_meta.shape)
            X_meta_test.append(x_meta)
        
        X_meta_test = np.hstack(X_meta_test)
        print(X_meta_test.shape)
        return self.meta_model.predict_proba(X_meta_test)
