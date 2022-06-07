import cv2
import numpy as np
import math

img_horizontal = cv2.imread('AFimg_horizontal.png',cv2.IMREAD_GRAYSCALE)
img_vertical = cv2.imread('AFimg_vertical.png',cv2.IMREAD_GRAYSCALE)

img_horizontal = np.float32(img_horizontal)
img_vertical = np.float32(img_vertical)

gradient = []
for i in range(img_horizontal.shape[0]):
    for j in range(img_horizontal.shape[1]):
       temp = ((img_horizontal[i][j])**2 + (img_vertical[i][j])**2)**(0.5)
       gradient.append(temp)
gradient = np.array(gradient)
gradient = np.uint8(gradient)
gradient = gradient.reshape(img_horizontal.shape[0],img_horizontal.shape[1])
print(gradient.shape)
cv2.imwrite('AF_gradient_img.png', gradient)