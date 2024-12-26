from .base import Model
from sklearn.linear_model import LogisticRegression
import joblib

class LogisticRegressionModel(Model):
    lr: LogisticRegression

    def __init__(self, model):
        self.lr = model

    @staticmethod
    def create(**kwargs) -> 'LogisticRegressionModel':
        C = kwargs.get('C', 1.0)
        max_iter = kwargs.get('max_iter', 10000)
        tol = kwargs.get('tol', 1e-4)
        return LogisticRegressionModel(
            LogisticRegression(
                penalty='l2', 
                C=C, 
                solver='saga', 
                max_iter=max_iter, 
                class_weight='balanced', 
                tol=tol
            )
        )
    
    def fit(self, X, y, **kwargs):
        self.lr.fit(X, y)

    def predict(self, X):
        return self.lr.predict(X)
    
    def predict_proba(self, X):
        return self.lr.predict_proba(X)
    
    def score(self, X, y):
        return self.lr.score(X, y)
    
    def save(self, path):
        joblib.dump(self.lr, path)

    @staticmethod
    def load(path) -> 'LogisticRegressionModel':
        return LogisticRegressionModel(joblib.load(path))
