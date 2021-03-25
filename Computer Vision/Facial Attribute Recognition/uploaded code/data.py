import cv2
import numpy as np
import os
import random
import config as cfg

path = 'D:\\BaiduNetdiskDownload\\CelebA\\Img\\img_align_celeba_png\\'
path_txt = 'D:\\BaiduNetdiskDownload\\CelebA\\Img\\train_40_att_list.txt'
feed_data = []
feed_label = []
feda = np.zeros(shape=(cfg.BATH_SIZE,224,224,3))
fela = np.zeros(shape=(cfg.BATH_SIZE,40))
f = open(path_txt,'r')
lines = f.readlines()
def get_data():
    while True:
        index = random.randint(0,len(lines)-1)
        name = lines[index].strip()
        filename = name.split(' ')[0]
        image = cv2.imread(path+filename)
        image = cv2.resize(image,(224,224))
        feed_data.append(image)
        feed_label.append(list(map(int,name.split(' ')[1:cfg.CLASSES+1])))
        #feed_label.append(int(name.split('_')[0]))
        if len(feed_data) == cfg.BATH_SIZE:
            feda[:,:,:,:] = feed_data
            fela[:,:] = feed_label
            yield feda,fela
            feed_data.clear()
            feed_label.clear()

#get_data()


