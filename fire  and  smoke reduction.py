import cv2
import numpy as np

# Load the image
image = cv2.imread('fire_input.jpg')
if image is None:
    print("Image not found!")
    exit()

# Resize for better viewing (optional)
image = cv2.resize(image, (640, 480))

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define fire color range in HSV
lower_fire = np.array([18, 150, 150])
upper_fire = np.array([35, 255, 255])

# Create a mask for fire-like colors
mask = cv2.inRange(hsv, lower_fire, upper_fire)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=4)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fire_found = False
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1500:
        fire_found = True
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "FIRE DETECTED", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display result
if fire_found:
    print("Fire detected in image!")
else:
    print("No fire detected.")

cv2.imshow("Fire Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()