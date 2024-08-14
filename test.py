import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split


# Load the image
image_path = 'data/Picture1.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.title('Original CFD Image')
plt.axis('off')
plt.show()

# Function to extract RGB values and convert to shear stress
def extract_shear_stress(image, scale_min, scale_max):
    # Define the color map scale (assuming a linear scale between min and max)
    # This depends on the color bar shown in the image.
    color_scale = np.linspace(scale_min, scale_max, num=256)
    
    # Extract the shear stress values based on the color mapping
    shear_stress = np.zeros((image.shape[0], image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            # Assuming 'pixel' is the index of the color scale
            shear_stress[i, j] = color_scale[pixel[0]]
    
    return shear_stress

# Example usage (assuming scale from 0 to 14)
shear_stress_data = extract_shear_stress(image, 0, 14)

# Display the extracted shear stress
plt.figure(figsize=(10, 8))
plt.imshow(shear_stress_data, cmap='viridis')
plt.title('Extracted Shear Stress Data')
plt.colorbar(label='Shear Stress')
plt.axis('off')
plt.show()



# Preparing the data for training
X = np.array([shear_stress_data])  # Input data (1 sample)
y = np.array([shear_stress_data])  # Target data (1 sample)

# Reshape for CNN input
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

# Splitting data (though we have one sample, this is just for example)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(X_train.shape[1] * X_train.shape[2], activation='linear'),
    # Reshape to match the output shape
    Dense(y_train.shape[1] * y_train.shape[2], activation='linear'),
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=1, validation_data=(X_test, y_test))

# Predictions
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(y_test.shape)

# Plotting the results
plt.figure(figsize=(20, 5))

# Original CFD
plt.subplot(1, 3, 1)
plt.imshow(y_test[0, :, :, 0], cmap='viridis')
plt.title('Original CFD')
plt.colorbar(label='Shear Stress')
plt.axis('off')

# CNN Prediction
plt.subplot(1, 3, 2)
plt.imshow(y_pred[0, :, :, 0], cmap='viridis')
plt.title('CNN Prediction')
plt.colorbar(label='Shear Stress')
plt.axis('off')

# Error Plot
error = np.abs(y_test[0, :, :, 0] - y_pred[0, :, :, 0])
plt.subplot(1, 3, 3)
plt.imshow(error, cmap='hot')
plt.title('Error Plot')
plt.colorbar(label='Error')
plt.axis('off')

plt.show()