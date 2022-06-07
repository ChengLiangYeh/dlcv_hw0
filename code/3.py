import cv2
import numpy as np
import math

img = cv2.imread('lena.png',cv2.IMREAD_GRAYSCALE)
print(img)

#sigma = 1 / (2 * math.log(2))
sigma = 20
print(sigma)
kernel = cv2.getGaussianKernel(3*3, sigma) #kernel size, sigma
kernel = kernel.reshape(3,3)
kernel = np.float32(kernel)


AFimg = cv2.filter2D(img, -1, kernel)
print(AFimg)
#cv2.imwrite('AFimg.png',AFimg)
cv2.imwrite('XXX.png',AFimg)
