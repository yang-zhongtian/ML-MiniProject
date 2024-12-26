from .base import Model
from sklearn.svm import SVC
import joblib

class SVMModel(Model):
    svm: SVC

    def __init__(self, model):
        self.svm = model

    @staticmethod
    def create(**kwargs):
        C = kwargs.get('C', 1.0)
        random_state = kwargs.get('random_state', 42)
        return SVMModel(SVC(C=C, random_state=random_state, probability=True))
    
    def fit(self, X, y, **kwargs):
        self.svm.fit(X, y)

    def predict(self, X):
        return self.svm.predict(X)
    
    def predict_proba(self, X):
        return self.svm.predict_proba(X)
    
    def score(self, X, y):
        return self.svm.score(X, y)
    
    def save(self, path):
        joblib.dump(self.svm, path)

    @staticmethod
    def load(path) -> 'SVMModel':
        return SVMModel(joblib.load(path))
