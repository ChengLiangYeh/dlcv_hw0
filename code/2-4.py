import numpy as np
import os
import cv2 
from sklearn.decomposition import PCA

path = os.listdir(os.getcwd())
#print(path[3])
os.chdir('./' + '%s' %path[3] )
#print(os.getcwd())
files = os.listdir(os.getcwd())
#print(files)
files.sort(key=lambda x:int(x.split('_')[0]))
print(files)
#此處讀取需要注意如果新增資料要改動!

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

training_data = readalldata(files)
print(training_data.shape)
training_label = []
for j in range(40):
    for i in range(9):
        training_label.append(j+1)
training_label=np.array(training_label)
print(training_label.shape)

###
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
myPCA = PCA()
myKNN = KNeighborsClassifier()
pipe = Pipeline(steps=[('myPCA', myPCA), ('myKNN', myKNN)])
print(pipe.steps[0])
print(pipe.steps[1])

param_grid = {
    'myPCA__n_components': [3,50,170],
    'myKNN__n_neighbors': [1,3,5],
}

grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=3)
grid_search.fit(training_data, training_label)
#print(grid_search.cv_results_)
import pandas as pd
df = pd.DataFrame(grid_search.cv_results_)
df.to_csv("2-4_data.csv")

