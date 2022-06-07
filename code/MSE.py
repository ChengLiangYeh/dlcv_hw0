import numpy as np
import cv2
from sklearn.metrics import mean_squared_error

y_data = cv2.imread('2_1.png')
#print(y_data)
y_data_flatten = y_data.reshape(1,56*46*3)

x_data = cv2.imread('RI_360_newer.png')
x_data_flatten = x_data.reshape(1,56*46*3)
error = mean_squared_error(y_data_flatten, x_data_flatten)
print(error)

#結果備註:
#3:93.49961180124224
#50:66.02639751552795
#170:36.027562111801245
#240:13.564052795031056
#345:0.578027950310559
#359:0.5108695652173914
#360:0.51074016563147

#問題:為何無法達到mse=0???