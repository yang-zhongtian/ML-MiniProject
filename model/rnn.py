from .base import Model
import os
os.environ["KERAS_BACKEND"] = "torch"
from keras import Input, Sequential, layers, optimizers, Model as KerasModel, callbacks

class RNNModel(Model):
    model: KerasModel
    history: callbacks.History = None

    def __init__(self, model):
        self.model = model

    @staticmethod
    def create(**kwargs) -> 'RNNModel':
        X_train_shape = kwargs.get('X_train_shape')
        y_train_len = kwargs.get('y_train_len')

        # Define the RNN model using Input layer
        model = Sequential()

        # Input layer
        model.add(Input(shape=X_train_shape))

        # Convolutional layer block 1
        model.add(layers.Conv1D(filters=64, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.3))

        # Convolutional layer block 2
        model.add(layers.Conv1D(filters=128, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.3))

        # Convolutional layer block 3
        model.add(layers.Conv1D(filters=256, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.3))

        # Recurrent layer
        model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)))

        # Fully connected layer
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))

        # Output layer
        model.add(layers.Dense(y_train_len, activation='softmax'))

        # Compile the model using sparse categorical crossentropy
        model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return RNNModel(model)

    
    def fit(self, X, y, **kwargs):
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 128)
        validation_data = kwargs.get('validation_data', None)
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.evaluate(X, y)
    
    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path) -> 'RNNModel':
        from keras import models
        model = models.load_model(path)
        return RNNModel(model)
