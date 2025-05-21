from turtle import forward
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
print(df.head())

print(df.shape)

df.drop(columns=['id', 'Unnamed: 32'], inplace = True)
print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encoding labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Convert numpy arrays to pytorch tensors
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)
print(X_train_tensor.shape)

# Define the model
class BreastCancerModel():
    def __init__(self, X):
        # Since the input data has 30 columns, we have 30 weights
        # Our weight matrix is a 30 x 1 matrix
        self.weights = torch.rand(X.shape[1], 1, dtype = torch.float64, requires_grad = True)
        self.bias = torch.zeros(1, dtype = torch.float64, requires_grad = True)

    def forward(self, X):
        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred

    def loss_function(self, y_pred, y_true):
        # Clamp predictions to avoid log(0)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

        # Binary Cross Entropy Loss
        loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return loss

# Parameters
learning_rate = 0.1
epochs = 25

# Training Pipeline
# 1. Forward pass
# 2. Compute loss
# 3. Backward pass
# 4. Update parameters

model = BreastCancerModel(X_train_tensor)

for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(X_train_tensor)
    
    # Compute loss
    loss = model.loss_function(y_pred, y_train_tensor)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    # Backward pass
    loss.backward()

    # Update parameters
    # Use torch.no_grad() to prevent tracking history
    # and to avoid gradients being calculated for the update step
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad

    # Set gradients to zero after updating
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    # Print loss in each epoch
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Evaluate the model
with torch.no_grad():
    y_test_pred = model.forward(X_test_tensor)
    y_test_pred = (y_test_pred > 0.9).float()  # Convert probabilities to binary predictions
    
    accuracy = (y_test_pred == y_test_tensor).float().mean()
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')