from __future__ import division, print_function, absolute_import
import tflearn
import numpy as np
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import image_preloader


dataset_file = 'C:\\Users\\Asad\\Desktop\\aeroplane_train.txt'
X, Y = image_preloader(dataset_file, image_shape=(64, 64),   mode='file', categorical_labels=True,   normalize=True)

X, Y = shuffle(X, Y)



img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()


img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()


x = tflearn.input_data(shape=[None, 64, 64, 3], name='input',data_preprocessing=img_prep,data_augmentation=img_aug)

x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_1')
x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
x = tflearn.dropout(x, 0.5, name='dropout1')

x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
x = tflearn.dropout(x, 0.5, name='dropout2')

x = tflearn.fully_connected(x, 2, activation='softmax', scope='fc8')
x = regression(x, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(x,tensorboard_verbose=0)
model.fit({'input': X}, {'targets': Y},shuffle=True, n_epoch=10,
            snapshot_step=500, show_metric=True, batch_size=100, run_id='aeroplane')
model.save('aeroplane_classification.model')


