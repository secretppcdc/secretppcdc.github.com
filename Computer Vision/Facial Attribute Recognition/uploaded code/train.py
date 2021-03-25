from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta,Adagrad,RMSprop,SGD
import numpy as np
import data
import config as cfg

seed = 7
np.random.seed(seed)

model = Sequential()
model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(224, 224, 3), padding='same', activation='relu',kernel_initializer='uniform',name='block1_conv1'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block1_conv2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block1_pool'))

model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block2_conv1'))
model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block2_conv2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block2_pool'))

model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block3_conv1'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block3_conv2'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block3_conv3'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block3_pool'))

model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block4_conv1'))
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block4_conv2'))
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block4_conv3'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block4_pool'))

model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block5_conv1'))
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block5_conv2'))
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform',name='block5_conv3'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block5_pool'))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(cfg.CLASSES, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adadelta(lr=0.005), metrics=['accuracy'])
model.summary()
#model.load_weights('./weight/vgg16_weights.h5',True)

generator = data.get_data()
model.fit_generator(generator,steps_per_epoch=cfg.STEPS_PER_EPOCH,epochs=cfg.EPOCH)

model.save('./weight/model.h5')