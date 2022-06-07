import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def show_img(img):
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Kmeans(after_preprocessing_img, clusters_K):
    #stop condition
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    sum_of_L2distance_from_each_point_to_their_centers, labels, centers = cv2.kmeans(after_preprocessing_img, clusters_K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return sum_of_L2distance_from_each_point_to_their_centers, labels, centers

def add_xy_features(img):
    #add more features like x, y 
    img = cv2.imread('bird.jpg')
    #print(img)
    #print(img.shape)
    index = np.arange(1024)
    #print(index)
    feature_map_x = np.vstack((index, index))
    for i in range(1022):
        feature_map_x = np.vstack((feature_map_x, index))
    #print(feature_map_x)
    feature_map_x = feature_map_x.reshape(1024,1024,1)
    #print(feature_map_x.shape)
    feature_map_y = feature_map_x.transpose()
    #print(feature_map_y)
    feature_map_y = feature_map_y.reshape(1024,1024,1)
    #print(feature_map_y.shape)
    img_combine_features = np.dstack((img, feature_map_x))
    img_combine_features = np.dstack((img_combine_features, feature_map_y))
    #print(img_combine_features.shape)
    return img_combine_features

'''
#第一題

#img preprocessing
img = cv2.imread('bird.jpg')
print(img.shape)
#show_img(img)
#reshape
flatten_img = img.reshape((-1,3))
print(flatten_img.shape)
print(flatten_img)
print(type(flatten_img[0,0]))
#change data type 
flatten_img = np.float32(flatten_img)

#clusters_K = 2,4,8,16,32
for i in range(5):
    clusters_K = 2**(i+1)
    print('clusters_K=',clusters_K)
    sum_of_L2distance_from_each_point_to_their_centers, labels, centers = Kmeans(flatten_img, clusters_K)
    print(sum_of_L2distance_from_each_point_to_their_centers)
    print(labels)
    print(centers)
    #post processing
    centers = np.uint8(centers)
    print(centers)
    print(labels.flatten())
    res = centers[labels.flatten()] #labels 是 0,1,2,3,4......看分幾類，然後丟回centers裡，意思是例如: labels=0那一類的pixel value就是centers 第0個row的那個center的pixel value
    print(res.shape)
    result_image = res.reshape(img.shape)
    show_img(result_image)
    #cv2.imwrite('clusters_K=%s.png' % clusters_K, result_image)
'''

'''
#第二題
img = cv2.imread('bird.jpg')
img_combine_features = add_xy_features(img)
flatten_img_CF = img_combine_features.reshape((-1,5))
print(flatten_img_CF.shape)
flatten_img_CF = np.float32(flatten_img_CF)
for i in range(5):
    clusters_K = 2**(i+1)
    print('clusters_K=',clusters_K)
    sum_of_L2distance_from_each_point_to_their_centers, labels, centers = Kmeans(flatten_img_CF, clusters_K)
    #print(centers)
    print(labels)
    #post processing
    centers = np.uint8(centers)
    print(centers)
    print(centers.shape[0])
    centers_onlyRGB = centers[0:(centers.shape)[0],0:3]
    print(centers_onlyRGB)
    res = centers_onlyRGB[labels.flatten()]
    result_image = res.reshape(img.shape)
    show_img(result_image)
    cv2.imwrite('old_RBGXY_clusters_K=%s.png' % clusters_K, result_image)
'''
'''
#效果不是很好=>測試data normalization
img = cv2.imread('bird.jpg')
img_B = img[0:1024,0:1024,0]
img_G = img[0:1024,0:1024,1]
img_R = img[0:1024,0:1024,2]
#opencv是BGR排列
    #img_B_scaled = preprocessing.scale(img_B)
    #dst = np.zeros(img_B.shape)
    #img_B_cv2_scaled = np.float32(cv2.normalize(img_B, dst, 1.0, 0.0, cv2.NORM_MINMAX))
#print(img_B)
img_B_scaled = np.float32(img_B/255)
#print(img_B_scaled)
img_G_scaled = np.float32(img_G/255)
img_R_scaled = np.float32(img_R/255)
img_B_scaled=img_B_scaled.reshape(1024,1024,1)
img_G_scaled=img_G_scaled.reshape(1024,1024,1)
img_R_scaled=img_R_scaled.reshape(1024,1024,1)
#print(img_R_scaled.shape)
index = np.arange(1024)
feature_map_x = np.vstack((index, index))
for i in range(1022):
    feature_map_x = np.vstack((feature_map_x, index))
feature_map_x = feature_map_x.reshape(1024,1024,1)
feature_map_y = feature_map_x.transpose()
feature_map_y = feature_map_y.reshape(1024,1024,1)
feature_map_x=feature_map_x/1023
feature_map_y=feature_map_y/1023
#print(feature_map_x)
#疊起來BGRXY
img_scaled_RGBXY = np.dstack((img_B_scaled,img_G_scaled))
img_scaled_RGBXY = np.dstack((img_scaled_RGBXY,img_R_scaled))
img_scaled_RGBXY = np.dstack((img_scaled_RGBXY,feature_map_x))
img_scaled_RGBXY = np.dstack((img_scaled_RGBXY,feature_map_y))
print(img_scaled_RGBXY.shape)
flatten_img_CF = img_scaled_RGBXY.reshape(img_scaled_RGBXY.shape[0]*img_scaled_RGBXY.shape[1],5)
print(flatten_img_CF.shape)
flatten_img_CF = np.float32(flatten_img_CF)
for i in range(5):
    clusters_K = 2**(i+1)
    print('clusters_K=',clusters_K)
    sum_of_L2distance_from_each_point_to_their_centers, labels, centers = Kmeans(flatten_img_CF, clusters_K)
    print(centers)
    centers_onlyRGB = centers[0:(centers.shape)[0],0:3]
    centers_onlyRGB = centers_onlyRGB*255
    centers_onlyRGB = np.uint8(centers_onlyRGB)
    print(centers_onlyRGB)
    print(labels)
    #post processing
    res = centers_onlyRGB[labels.flatten()]
    result_image = res.reshape(img.shape)
    show_img(result_image)
    cv2.imwrite('RGBXY_clusters_K=%s.png' % clusters_K, result_image)
'''
