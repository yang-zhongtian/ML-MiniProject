from model.base import Model
import numpy as np

class Evaluator:
    model: Model
    X_test: np.ndarray
    y_test: np.ndarray

    def __init__(self, model: Model, X_test: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    
    @property
    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        y_pred = self.model.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)

    def plot_confusion_matrix(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        class_labels = ['Positive', 'Negative']
        cm = pd.DataFrame(self.confusion_matrix, index=class_labels, columns=class_labels)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()
    
    @property
    def roc_auc(self) -> tuple[np.ndarray, np.ndarray, float]:
        from sklearn.metrics import roc_curve, auc
        y_pred = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    
    def plot_roc_auc(self):
        import matplotlib.pyplot as plt
        fpr, tpr, roc_auc = self.roc_auc
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-specificity')
        plt.ylabel('sensitivity')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
