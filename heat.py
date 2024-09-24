import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from sklearn.model_selection import train_test_split

# Load the image
image_path = 'images/test1.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.title('Original CFD Image')
plt.axis('off')
plt.show()

# Function to extract RGB values and convert to shear stress
def extract_shear_stress(image, colormap):
    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Normalize to [0, 1] range
    gray_image = gray_image / 255.0
    
    # Convert grayscale to shear stress values based on colormap
    shear_stress = colormap(gray_image)[:, :, 0]  # Use colormap to get the stress values

    return shear_stress

# Define a colormap (for example, viridis)
colormap = plt.get_cmap('viridis')
shear_stress_data = extract_shear_stress(image, colormap)

# Display the extracted shear stress
plt.figure(figsize=(10, 8))
plt.imshow(shear_stress_data, cmap='viridis')
plt.title('Extracted Shear Stress')
plt.axis('off')
plt.show()

# Preparing the data for training
X = np.array([shear_stress_data])  # Input data (1 sample)
y = np.array([shear_stress_data])  # Target data (1 sample)

# Reshape for CNN input
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
# Skip train-test split
X_train, X_test = X, X
y_train, y_test = y, y

# CNN Model (rest of the code remains the same)
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    UpSampling2D(size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    UpSampling2D(size=(2, 2)),
    Conv2D(1, kernel_size=(3, 3), activation='linear', padding='same'),
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=1, validation_data=(X_test, y_test))

# Predictions
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(y_test.shape)

# Plotting the results
plt.figure(figsize=(20, 5))

# Original CFD
plt.subplot(1, 4, 1)
plt.imshow(y_test[0, :, :, 0], cmap='viridis')
plt.title('Original CFD')
plt.colorbar(label='Shear Stress')
plt.axis('off')

# CNN Prediction
plt.subplot(1, 4, 2)
plt.imshow(y_pred[0, :, :, 0], cmap='viridis')
plt.title('CNN Prediction')
plt.colorbar(label='Shear Stress')
plt.axis('off')

# Error Plot
error = np.abs(y_test[0, :, :, 0] - y_pred[0, :, :, 0])
plt.subplot(1, 4, 3)
plt.imshow(error, cmap='hot')
plt.title('Error Plot')
plt.colorbar(label='Error')
plt.axis('off')

# Error Graph
plt.subplot(1, 4, 4)
plt.plot(error.flatten())
plt.title('Error Values')
plt.xlabel('Pixel Index')
plt.ylabel('Error')

plt.tight_layout()
plt.show()