# -*- coding: utf-8 -*-
# @Time    : 2023/2/19 21:09
# @Author  : zwl
import os

import keras
import numpy
import numpy as np
from keras import Model
from keras.models import Sequential
from keras.regularizers import L1
from tensorflow.python.keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt

from tools.data_process import AmendCFO
from tools.plot_picture import DestinyFig
from .CFOMendModel1 import CFOMendModel
import tensorflow as tf
import pickle as pkl


class CLDNNWithCFOMend(Model):
    def __init__(self, num_classes, IQShape, sampRate, subWeights):
        super(CLDNNWithCFOMend, self).__init__()

        def cal1(cfo):
            t = tf.constant(np.linspace(0, 1 / float(sampRate) * IQShape[0], IQShape[0]).astype(np.float32))
            return tf.keras.backend.cos(2 * np.pi * cfo * t)

        def cal2(cfo):
            t = tf.constant(np.linspace(0, 1 / float(sampRate) * IQShape[0], IQShape[0]).astype(np.float32))
            return tf.keras.backend.sin(2 * np.pi * cfo * t)

        self.CFOMendModel = CFOMendModel(IQShape)
        self.CFOMendModel.load_weights(subWeights)
        # for layer in self.CFOMendModel.layers:
        #     layer.trainable = False

        self.cos = tf.keras.layers.Lambda(cal1, name="deal1")
        self.sin = tf.keras.layers.Lambda(cal2, name="deal2")
        self.x11 = tf.keras.layers.Multiply()
        self.x12 = tf.keras.layers.Multiply()
        self.x21 = tf.keras.layers.Multiply()
        self.x22 = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()
        self.sub = tf.keras.layers.Subtract()
        self.reshape1 = tf.keras.layers.Reshape(target_shape=((IQShape[0], 1)))
        self.reshape2 = tf.keras.layers.Reshape(target_shape=((IQShape[0], 1)))

        self.CLDNN = Sequential([
            tf.keras.layers.Input(shape=IQShape),
            tf.keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.MaxPool1D(pool_size=2),
            # tf.keras.layers.LSTM(16, return_sequences=True, ),
            # tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(64, return_sequences=True, ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, x, **kwargs):
        cfo = self.CFOMendModel(x[0])
        cos = self.cos(cfo)
        sin = self.sin(cfo)
        x11 = self.x11([x[1], cos])
        x12 = self.x12([x[2], sin])
        x21 = self.x21([x[2], cos])
        x22 = self.x22([x[1], sin])
        yi = self.add([x11, x12])
        yq = self.sub([x21, x22])
        yi = self.reshape1(yi)
        yq = self.reshape2(yq)
        amendData = tf.keras.layers.Concatenate(axis=2)([yi, yq])
        out = self.CLDNN(amendData)
        return out

    # def call(self, x, **kwargs):
    #     cfo = self.CFOMendModel(x[0])
    #     cos = self.cos(cfo)
    #     sin = self.sin(cfo)
    #     x11 = self.x11([x[1], cos])
    #     x12 = self.x12([x[2], sin])
    #     x21 = self.x21([x[2], cos])
    #     x22 = self.x22([x[1], sin])
    #     yi = self.add([x11, x12])
    #     yq = self.sub([x21, x22])
    #     yi = self.reshape1(yi)
    #     yq = self.reshape2(yq)
    #     amendData = tf.keras.layers.Concatenate(axis=2)([yi, yq])
    #     return amendData


if __name__ == "__main__":
    model = CLDNNWithCFOMend(11, (1024, 2))
    # model.build(input_shape=[(None, 1024, 2), (None, 1024), (None, 1024)])
    # model.summary()
    # plot_model(model, to_file='model1.png', show_shapes=True)  # print model

    dataset = 'usrp'
    amendCFO = AmendCFO(31, 0.25)
    fig_path = '../../results/cldnn_with_cfo_mend/' + dataset
    CFOMendModel = CFOMendModel((1024, 2))
    CFOMendModel.load_weights(
        r'E:/Users/mycode/pycharmprojects/AMR_CFO/models_checkpoint/CFOMendModel/usrp/0.001/weight')

    if dataset == 'usrp':
        DataSavePath = '../../data/' + dataset + '/origin/realData_CFO0.001/1024/picData_1024_realData_CFO0.001.pkl'
        with open(DataSavePath, 'rb') as f:
            picData = pkl.load(f)
        choiceMod = set(["4ask", "qpsk", "bpsk", "8psk", "16qam", "64qam"])
        for sampleName, data in picData.items():
            mod = sampleName.split('_')[0]
            if mod not in choiceMod:
                continue
            x = np.transpose([np.real(data), np.imag(data)])
            x = np.array([x, ])
            y = model.predict((x, x[:, :, 0], x[:, :, 1]))
            # freq = CFOMendModel.predict(x)[0]
            # cos = np.cos(2 * np.pi * freq * np.linspace(0, realData_CFO0.005 / float(0.2e6) * len(data), len(data)).astype(np.float32))
            # sin = np.sin(2 * np.pi * freq * np.linspace(0, realData_CFO0.005 / float(0.2e6) * len(data), len(data)).astype(np.float32))
            # amendData = np.real(data) * cos + np.imag(data) * sin + 1j * (np.imag(data) * cos - np.real(data) * sin)
            # amendData = data * np.exp(
            #     1j * 2 * np.pi * -freq * np.linspace(0, realData_CFO0.005 / float(0.2e6) * len(data), len(data)))
            plt.rcParams['figure.figsize'] = (105 / 100, 105 / 100)
            plt.rcParams['savefig.dpi'] = 100
            path = os.path.join(fig_path, mod)
            if not os.path.exists(path):
                os.makedirs(path)
            DestinyFig(x[0][:, 0], x[0][:, 1], path, str(sampleName.split('_')[1]) + 'before').plot_destiny()
            # DestinyFig(np.real(amendData), np.imag(amendData), path,
            #            str(sampleName.split('_')[realData_CFO0.005]) + 'after_manual').plot_destiny()
            DestinyFig(y[0][:, 0], y[0][:, 1], path, str(sampleName.split('_')[1]) + 'after').plot_destiny()
