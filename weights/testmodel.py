import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from glob import glob
from PIL import Image
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras import Sequential
from keras.models import load_model
from keras.layers import concatenate
from keras.layers import BatchNormalization, Activation, Conv2DTranspose
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

img_path = '/Users/weihaiyu/PycharmProjects/cv/Dataset/'
CLASS = 'Yes'
all_files = os.listdir(img_path + CLASS)
files = [item for item in all_files if "img" in item]
random.shuffle(files)
img_num = len(files)
target_size = (240,240)


def metric_fun(y_true, y_pred):
    fz = tf.reduce_sum(2 * y_true * tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-5
    fm = tf.reduce_sum(y_true + tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-5
    # print('------------')
    # print(fz/fm)
    return fz / fm

def load_data():
    x_train = []  # 定义一个空列表，用于保存数据集
    x_label = []
    for (n, file_name) in enumerate(files):
        img = np.array(np.load(os.path.join(img_path, CLASS, file_name)), dtype='float32')
        x_train.append(img)
    for (n, file_name) in enumerate(files):
        img = np.array(np.load(os.path.join(img_path, CLASS, file_name.split('_')[0] + '_seg.npy')), dtype='float32')
        x_label.append(img)

    # x_train = np.expand_dims(np.array(x_train), axis=3)  # 扩展维度，增加第4维
    # x_label = np.expand_dims(np.array(x_label), axis=3)  # 变为网络需要的输入维度(num, 256, 256, 1)
    x_train = np.array(x_train)
    x_label = np.array(x_label)
    print(x_label.shape)
    # print(test.shape)
    # print(test)
    np.random.seed(116)  # 设置相同的随机种子，确保数据匹配
    np.random.shuffle(x_train)  # 对第一维度进行乱序
    np.random.seed(116)
    np.random.shuffle(x_label)

    # print(x_label)
    # print(x_train)
    # 图片有三千张左右，按9:1进行分配
    return x_train[:2700, :, :], x_label[:2700, :, :], x_train[2700:, :, :], x_label[2700:, :, :]


model = load_model('best_model.h5', compile=False)
x_train, x_label, y_train, y_label = load_data()

test_num = y_train.shape[0]
for epoch in range(5):
    rand_index = []
    while len(rand_index) < 3:
        np.random.seed()
        temp = np.random.randint(0, test_num, 1)
        if np.sum(x_label[temp]) > 0:  # 确保产生有肿瘤的编号
            rand_index.append(temp)
    rand_index = np.array(rand_index).squeeze()
    fig, ax = plt.subplots(3, 3, figsize=(18, 18))
    for i, index in enumerate(rand_index):
        mask = model.predict(x_train[index:index + 1]) > 0.1
        ax[i][0].imshow(x_train[index].squeeze()[:,:,0], cmap='turbo')
        ax[i][0].set_title('network input', fontsize=20)
        # 计算dice系数
        fz = 2 * np.sum(mask.squeeze() * x_label[index].squeeze())
        fm = np.sum(mask.squeeze()) + np.sum(x_label[index].squeeze())
        dice = fz / fm
        ax[i][1].imshow(mask.squeeze()[:,:])
        ax[i][1].set_title('network output(%.4f)' % dice, fontsize=20)  # 设置title
        ax[i][2].imshow(x_label[index].squeeze()[:,:])
        ax[i][2].set_title('mask label', fontsize=20)
    fig.savefig('./evaluation/show%d_%d_%d.png' % (rand_index[0], rand_index[1], rand_index[2]),
                bbox_inches='tight', pad_inches=0.1)  # 保存绘制的图片
    print('finished epoch: %d' % epoch)
    plt.close()