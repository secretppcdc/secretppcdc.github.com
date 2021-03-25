import keras
import cv2
import os
import numpy as np

path_weight = './weight/model.h5'
path_image = 'D:\\BaiduNetdiskDownload\\CelebA\\Img\\test_40_att_list.txt'
path_pic = 'D:\\BaiduNetdiskDownload\\CelebA\\Img\\img_align_celeba_png\\'
thre = 0.7

feed_test = np.zeros(shape=(1,224,224,3))
model_vgg = keras.models.load_model(path_weight)
f = open('result_label.txt','w')

test_file = open(path_image,'r')
lines = test_file.readlines()
acc = [0 for i in range(40)]

for i in lines:
    name = i.split(' ')[0]
    print(name)
    label = i.strip().split(' ')[1:]
    image = cv2.imread(path_pic + name)
    image = cv2.resize(image, (224, 224))
    feed_test[0, :, :, :] = image
    result = model_vgg.predict(feed_test)[0]
    result_1 = ['1' if result[i] > thre else '0' for i in range(len(result))]
    for index,j in enumerate(label):
        if j == result_1[index]:
            acc[index] += 1
acc = [acc[index]/len(lines) for index in range(len(acc))]
val = sum(acc)/len(acc)
acc = list(map(str,acc))
f.write('Acc:'+' '.join(acc)+'\n')
f.write('AverAcc:'+str(val))
    #print(result_1)

# for i in os.listdir(path_image):
#     for j in os.listdir(path_image+i):
#         try:
#             image = cv2.imread(path_image+i+'/'+j)
#             image = cv2.resize(image,(224,224))
#             feed_test[0,:,:,:] = image
#             result = model_vgg.predict(feed_test)[0]
#             result_1 = ['1' if result[i] > thre else '-1' for i in range(len(result))]
#             f.write(j+' '+' '.join(result_1)+'\n')
#         except:
#             print(j)
#         #cv2.waitKey(10)
# f.close()
