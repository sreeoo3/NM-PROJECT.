import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image = cv2.imread('bgcherry.jpg')
background_image = cv2.imread('virtualbg.jpg')
background_image = cv2.resize(background_image, (input_image.shape[1], input_image.shape[0]))
hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
lower_bound = np.array([35, 40, 40])
upper_bound = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower_bound, upper_bound)
mask_inv = cv2.bitwise_not(mask)
foreground = cv2.bitwise_and(input_image, input_image, mask=mask_inv)
background = cv2.bitwise_and(background_image, background_image, mask=mask)
result = cv2.add(foreground, background)
cv2.imwrite('output.jpg', result)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Output Image')
plt.show()