import cv2
import numpy as np
import math

img = cv2.imread('AFimg.png',cv2.IMREAD_GRAYSCALE)
print(img)

kernelx = [[+1,0,-1],[+1,0,-1],[+1,0,-1]]
kernely = [[+1,+1,+1],[0,0,0],[-1,-1,-1]]
kernelx = np.array(kernelx)
kernely = np.array(kernely)

img_vertical = cv2.filter2D(img, -1, kernelx)
cv2.imwrite('AFimg_vertical.png', img_vertical)

img_horizontal = cv2.filter2D(img, -1, kernely)
cv2.imwrite('AFimg_horizontal.png', img_horizontal)