import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
import skorch.callbacks
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to create a model with specified parameters
def create_model(input_size, output_size):
    return NeuralNetClassifier(
        module=NeuralNetwork,
        module__input_size=input_size,
        module__hidden_size=100,  # Default which will be overwritten by hyperparameter search
        module__output_size=output_size,
        module__dropout_rate=0.5,  # Default
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,  # Default
        max_epochs=20,
        batch_size=32,
        iterator_train__shuffle=True,
        callbacks=[('early_stopping', skorch.callbacks.EarlyStopping(patience=5))]
    )

# Function to optimize hyperparameters using BayesSearchCV
def optimize_hyperparameters(X_train, y_train):
    model = create_model(input_size=X_train.shape[1], output_size=len(set(y_train)))
    search = BayesSearchCV(
        estimator=model,
        search_spaces={
            'optimizer__lr': Real(1e-4, 1e-1, prior='log-uniform'),
            'module__hidden_size': Integer(32, 256),
            'module__dropout_rate': Real(0.1, 0.7),
        },
        n_iter=10,
        cv=3,  # Number of cross-validation folds
        verbose=1,
        n_jobs=1  # Set to 1 to avoid multiprocessing issues
    )
    search.fit(X_train.astype(np.float32), y_train)
    return search.best_params_

if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params = optimize_hyperparameters(X_train, y_train)
    print("Best Parameters:", best_params)