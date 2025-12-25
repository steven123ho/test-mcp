import cv2
import numpy as np

# Paths to the two images to compare
image1_path = 'image1.png'
image2_path = 'image2.png'
output_path = 'diff_output.png'

# Read the images
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Ensure the images have the same size
if img1.shape != img2.shape:
    raise ValueError('Images must have the same dimensions and channels')

# Compute the absolute difference
diff = cv2.absdiff(img1, img2)

# Convert the difference to grayscale
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to get the regions with significant differences
_, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

# Find contours of the differences
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around the differences on the first image
highlighted = img1.copy()
for contour in contours:
    if cv2.contourArea(contour) > 40:  # filter out small differences
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save the result
cv2.imwrite(output_path, highlighted)

print(f'Differences highlighted and saved to {output_path}') 