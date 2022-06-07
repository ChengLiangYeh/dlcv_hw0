import numpy as np
import os
import cv2 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

path = os.listdir(os.getcwd())
#print(path)
os.chdir('./' + '%s' %path[4] )
#print(os.getcwd())
files = os.listdir(os.getcwd())
#print(files)
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

training_data = readalldata(files)
#print(training_data)
#print(training_data.shape)

###Draw meanface 
meandata = np.mean(training_data,axis=0)
meandata = np.uint8(meandata)
meanface = meandata.reshape(56,46,3)
#cv2.imwrite('meanface.png', meanface)
###

#####scaler = StandardScaler()
#####scaler.fit(training_data)
#print('before=',training_data)
#####training_data = scaler.transform(training_data)
#print('after=',training_data)
#data標準化方法補充:https://blog.csdn.net/m0_47478595/article/details/106402843?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.edu_weight&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.edu_weight
pca = PCA(n_components=360)
transformed_data = pca.fit_transform(training_data)
#print("transformed_data.shape:", transformed_data.shape)
###觀察解釋變異率
'''
cum_explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cum_explained_var_ratio)
plt.xlabel('# principal components')
plt.ylabel('cumulative explained variance')
plt.show()
'''

#此處使用minmax轉換回去，Q:為何不用standard scaler轉換呢? ->用standard scaler轉換有問題!(結果論)
#print('pca.components_.shape=',pca.components_.shape)
#####scaled_comps = minmax_scale(pca.components_, axis=1) #注意此處eigenface要轉換回去成可讀的形式需使用minmax轉換，也就是第一題要使用standard scaler把資料做轉換跑pca，然後再用minmax轉換pca.components_回去成可讀的形式 -> eigenface
scaled_comps = pca.components_ #####
#print(scaled_comps.shape)
#print(scaled_comps)
for i in range(scaled_comps.shape[0]):
    scaled_comps_img = scaled_comps[i].reshape(56,46,3)
    print(scaled_comps_img)
    #####scaled_comps_img = np.uint8(scaled_comps_img*255)
    #cv2.imwrite('eigenface_%s.png' %(i+1), scaled_comps_img)

'''
#第二題注意: 跟第一題不同之處:不用過standard scaler，直接跑pca得到之pca.components_跟transformed_data做內積再加上meanface(沒過standard scaler) -> 重建之影像
#第二題
#eiganvector = pca.components_ (component數量,component長度)
#最大的eigenvalue = pca.explained_variance_
#RECONSTRUCT FACE WITH　COMPONENTS
#使用前n張eigenface(eigenvector)，n = 3、50、170、240、345
#scaled_comps是pca.components的rescale
#transformed_data是eigenvalue
#print(scaled_comps.shape)
#print(transformed_data.shape) #(360張img, 359根eigenvector)
#print(transformed_data[0,:].shape)
result = np.zeros([1,7728])
for i in range(360): #很笨的寫法，要自己改用多少根ｅｉｇｅｎｖｅｃｔｏｒｓ
    ###temp = np.dot( transformed_data[9, i], scaled_comps[i, :] )
    temp = np.dot( transformed_data[9, i], pca.components_[i, :] )
    result = result + temp

scaled_meandata = np.mean(training_data,axis=0) #training data 在上面有經過standard scaler再取平均 #更新:沒經過了

#result = result + meandata
result = result + scaled_meandata
#print(result.shape)
print(result)
#####result = minmax_scale(result, axis=1) #ｄａｔａ還原到０～１區間
#####result = np.uint8(result*255) # 0~1 -> 0~255
result = np.uint8(result)#####
Reconstructed_image = result.reshape(56,46,3)
cv2.imshow('IMAGE',Reconstructed_image,)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('RI_360.png', Reconstructed_image)
'''