import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Step 1: Load and Preprocess Data
def load_data():
    X = np.random.rand(100, 64, 64, 1)  # 100 samples of 64x64 grayscale images
    y = np.random.rand(100, 64, 64, 1)  # Corresponding stress maps with same shape as X
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# Debugging: Check the shapes of the data
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Step 2: Define the CNN Model
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(np.prod(input_shape[:-1]), activation='linear'),
        layers.Reshape(input_shape[:-1])  # Reshape to original image dimensions
    ])
    return model

input_shape = X_train.shape[1:]  # Should be (height, width, channels)
model = create_model(input_shape)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Step 3: Train the CNN Model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=16)

# Step 4: Evaluate the Model and Generate Predictions
test_loss, test_mae = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)

# Step 5: Visualize the Results
def plot_results(y_true, y_pred, index=0):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('CFD Contours')
    plt.contourf(y_true[index, :, :, 0], levels=100, cmap='jet')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title('CNN Predicted Contours')
    plt.contourf(y_pred[index, :, :, 0], levels=100, cmap='jet')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title('Error Map')
    plt.contourf(y_true[index, :, :, 0] - y_pred[index, :, :, 0], levels=100, cmap='jet')
    plt.colorbar()
    
    plt.show()

# Plot results for the first test sample
plot_results(y_test, predictions, index=0)
