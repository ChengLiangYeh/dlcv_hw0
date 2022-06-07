import numpy as np
import os
import cv2 
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#找出當前路徑，#此處讀取需要注意如果新增資料要改動!
path = os.listdir(os.getcwd())
os.chdir('./' + '%s' %path[4] )
files = os.listdir(os.getcwd())
files.sort(key=lambda x:int(x.split('_')[0]))
print(files)

def readalldata(filesnamelist):
    data = np.empty([1,7728],dtype='uint8') #7728 = image size*image channel
    #print(data)
    for i in range(len(filesnamelist)):
        img = cv2.imread(filesnamelist[i])
        #print('Read ' + filesnamelist[i])
        #print(img.shape)
        img_flatten = img.reshape(-1)
        #print(img_flatten.shape)
        data = np.vstack((data,img_flatten))
        #print(data.shape)
    data = data[1:data.shape[0],:]
    #print(data.shape)
    #print(data)
    return data

#製作training data
training_data = readalldata(files)
print(training_data.shape)
training_label = []
for j in range(40):
    for i in range(9):
        training_label.append(j+1)
training_label=np.array(training_label)
print(training_label.shape)

#切換路徑至testing data資料夾
os.chdir('C:\\Users\\showt\\Desktop\\DLCV_new\\p2_data')
path = os.listdir(os.getcwd())
os.chdir('./' + '%s' %path[3] )   #先print出來才知道是3
files = os.listdir(os.getcwd())
files.sort(key=lambda x:int(x.split('_')[0]))
print(files)

#製作testing data
testing_data = readalldata(files)
print(testing_data.shape)
testing_label = []
for i in range(40):
    testing_label.append(i+1)
testing_label=np.array(testing_label)
print(testing_label.shape)

#使用2-4題grid search 找到的hyperparameter n = 170 , k = 1
n = 170
k = 1
myPCA = PCA(n)
myPCA.fit(training_data)
#PCA降維
DR_training_data = myPCA.transform(training_data)
DR_testing_data = myPCA.transform(testing_data)
print(DR_training_data.shape)
print(DR_testing_data.shape)
#train KNN 
myKNN = KNeighborsClassifier(k)
myKNN.fit(DR_training_data, training_label)
#test KNN
predict_result = myKNN.predict(DR_testing_data)
accuracy = metrics.accuracy_score(testing_label, predict_result)
print('accuracy=', accuracy)

#備註: final accuracy = 0.95!