from abc import ABC, abstractmethod

class Model(ABC):
    @staticmethod
    @abstractmethod
    def create(**kwargs) -> 'Model':
        pass

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def score(self, X, y):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    @abstractmethod
    def load(path) -> 'Model':
        pass