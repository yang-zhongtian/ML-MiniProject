from .base import Model
from sklearn.ensemble import RandomForestClassifier
import joblib

class RandomForestModel(Model):
    rf: RandomForestClassifier

    def __init__(self, model):
        self.rf = model

    @staticmethod
    def create(**kwargs) -> 'RandomForestModel':
        n_estimators = kwargs.get('n_estimators', 100)
        random_state = kwargs.get('random_state', 42)
        return RandomForestModel(RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
    
    def fit(self, X, y, **kwargs):
        self.rf.fit(X, y)

    def predict(self, X):
        return self.rf.predict(X)
    
    def predict_proba(self, X):
        return self.rf.predict_proba(X)
    
    def score(self, X, y):
        return self.rf.score(X, y)
    
    def save(self, path):
        joblib.dump(self.rf, path)

    @staticmethod
    def load(path) -> 'RandomForestModel':
        return RandomForestModel(joblib.load(path))
