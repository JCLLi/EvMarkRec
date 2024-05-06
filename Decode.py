import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread("./Test/merge.PNG")
# img = cv2.imread("./Test/merge.PNG", cv2.IMREAD_GRAYSCALE)

# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#
# contours, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
#
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
#
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Threshold the image to create a binary image
# _, thresh = cv2.threshold(gray, 126, 200, cv2.THRESH_BINARY)
#
# # Find contours in the binary image
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Draw contours on the original image
# cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
#
# # Display the result
# cv2.imshow('Contours', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


image = cv2.imread("./Test/merge.PNG")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Harris Corner Detection
harris_response = cv2.cornerHarris(gray_blurred, 2, 3, 0.04)

# Threshold to highlight corners
threshold = 0.01 * harris_response.max()
corner_image = np.zeros_like(image)
corner_image[harris_response > threshold] = [0, 0, 255]

# Display the detected corners on the original image
image_with_corners = cv2.add(image, corner_image)

# Convert BGR to RGB for Matplotlib
image_with_corners_rgb = cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB)

# Show the result using Matplotlib
plt.imshow(image_with_corners_rgb)
plt.axis('off')
plt.title('Corners Detected')
plt.show()