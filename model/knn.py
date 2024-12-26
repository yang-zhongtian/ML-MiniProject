from .base import Model
from sklearn.neighbors import KNeighborsClassifier
import joblib

class KNNModel(Model):
    knn: KNeighborsClassifier

    def __init__(self, model):
        self.knn = model

    @staticmethod
    def create(**kwargs) -> 'KNNModel':
        k = kwargs.get('k', 5)
        return KNNModel(KNeighborsClassifier(n_neighbors=k))
    
    def fit(self, X, y, **kwargs):
        self.knn.fit(X, y)

    def predict(self, X):
        return self.knn.predict(X)
    
    def predict_proba(self, X):
        return self.knn.predict_proba(X)
    
    def score(self, X, y):
        return self.knn.score(X, y)
    
    def save(self, path):
        joblib.dump(self.knn, path)

    @staticmethod
    def load(path) -> 'KNNModel':
        return KNNModel(joblib.load(path))