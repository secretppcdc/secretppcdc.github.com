import keras
import cv2
import os
import numpy as np

path = './weight/model.h5'
path_image = './testset/'
thre = 0.2

feed_test = np.zeros(shape=(1,224,224,3))
model_vgg = keras.models.load_model(path)
f = open('predictions.txt','w')
for i in os.listdir(path_image):
    for j in os.listdir(path_image+i):
        try:
            image = cv2.imread(path_image+i+'/'+j)
            image = cv2.resize(image,(224,224))
            feed_test[0,:,:,:] = image
            result = model_vgg.predict(feed_test)[0]
            result_1 = ['1' if result[i] > thre else '-1' for i in range(len(result))]
            f.write(j+' '+' '.join(result_1)+'\n')
        except:
            print(j)
        #cv2.waitKey(10)
f.close()
