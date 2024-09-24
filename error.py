import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the original and created images
original_img = Image.open('1.png')
created_img = Image.open('2.png')

# Ensure both images are of the same size by resizing created_img
created_img_resized = created_img.resize(original_img.size)

# Convert images to grayscale and to numpy arrays
original_array = np.array(original_img.convert('L'))
created_array = np.array(created_img_resized.convert('L'))

# Ensure both images are of the same size
if original_array.shape != created_array.shape:
    raise ValueError("The images do not have the same dimensions!")

# Calculate the absolute difference between the original and created image
difference = np.abs(original_array - created_array)

# Calculate the error as a percentage
error_percentage = (difference / 255.0) * 100

# Calculate the mean error
mean_error = np.mean(error_percentage)

# Plotting the error
plt.figure(figsize=(10, 5))
plt.plot(error_percentage.flatten(), label='Error')
# plt.axhline(y=5, color='r', linestyle='--', label='5% threshold')
plt.title(f'Mean Error: {mean_error:.2f}%')
plt.xlabel('Pixel Index')
plt.ylabel('Error Percentage')
plt.legend()
plt.show()

# Check if the mean error is less than 5%
if mean_error < 5:
    print(f"Success! The mean error is {mean_error:.2f}%, which is less than 5%.")
else:
    print(f"Warning! The mean error is {mean_error:.2f}%, which exceeds the 5% threshold.")
