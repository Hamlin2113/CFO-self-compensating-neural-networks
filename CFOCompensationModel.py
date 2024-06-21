import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import *
import math
from keras import Model
import pickle as pkl
from tools.data_process.AmendCFO import AmendCFO
from tools.data_process.DataSetSplit import train_valid_test_split
from tools.plot_picture.DestinyFig import DestinyFig
import os
import pandas as pd

num_epochs = 100
batch_size = 50
learning_rate = 0.001
alpha = 0.1


# Embedding of empirical loss in data input
class CFOMendModel(Model):
    def __init__(self, dim):
        super(CFOMendModel, self).__init__()
        self.cfomendmodel = Sequential([
            tf.keras.layers.Input(shape=dim),
            tf.keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu'),
            # tf.keras.layers.LSTM(64, return_sequences=True, ),
            # tf.keras.layers.GRU(units=64),
            # tf.keras.layers.LayerNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(dim[0]),
            tf.keras.layers.Dense(int(dim[0] / 2)),
            tf.keras.layers.Dense(1)
        ])
        self.val_loss = math.inf

    def call(self, input, **kwargs):
        return self.cfomendmodel(input)

    # Empirical function plus regularization term
    def get_loss(self, input, empirical_label, sampleRate):
        output = self.cfomendmodel(input)
        val1 = [label[0] for label in empirical_label]
        val2 = [label[1] for label in empirical_label]
        loss1 = tf.reduce_mean(tf.losses.mean_squared_error(output, val1))
        loss2 = tf.reduce_mean(tf.losses.mean_squared_error(output, val2))
        # output = output.numpy()
        # amend = AmendCFO(31, 0.25)
        # amendDatas = [(input[i][:, 0] + 1j * input[i][:, 1]) * np.exp(
        #     1j * 2 * np.pi * -output[i] * np.linspace(0, 1 / float(sampleRate) * len(input[i]), len(input[i])))
        #               for i in range(len(input))]
        # ind = np.average(
        #     [amend.indic(amend.count_sum_in_grid(iq_to_polar(amendData))) for amendData in amendDatas]) * 100
        return tf.math.minimum(loss1, loss2)
        # return loss1 + ind

    def get_grad_update(self, input, empirical_label, sampleRate):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            self.train_loss = self.get_loss(input, empirical_label, sampleRate)
        g = tape.gradient(self.train_loss, self.variables)
        # optimizers.Adam(learning_rate=learning_rate, batch_size=batch_size).apply_gradients( grads_and_vars=zip(g, self.variables))
        optimizers.Adam(learning_rate=learning_rate).apply_gradients(grads_and_vars=zip(g, self.variables))
        return self.train_loss

    def get_val_loss(self, input, empirical_label, sampleRate, file_name):
        val_loss = self.get_loss(input, empirical_label, sampleRate)
        if val_loss.numpy() < self.val_loss:
            self.val_loss = val_loss.numpy()
            self.save_weights(file_name)
        return val_loss, self.val_loss

    def getCFO(self, input):
        return self.cfomendmodel(input)


if __name__ == '__main__':
    dataset = 'usrp'
    sampLen = 1024
    model = CFOMendModel((sampLen, 2))
    # model.build(input_shape=(None, 128, 2))
    # model.summary()
    # plot_model(model, to_file='model1.png', show_shapes=True)  # print model
    amendCFO = AmendCFO(31, 0.25)

    if dataset == 'usrp':
        model_path = '../../models_checkpoint/CFOMendModel/origin/realData_2.4G_1M_CFO_0-10k/model_weight'
        DataSavePath = '../../data/usrp/origin/realData_2.4G_1M_CFO_0-10k/picData_1024_realData_2.4G_1M_CFO_0-10k.pkl'
        with open(DataSavePath, 'rb') as f:
            picData = pkl.load(f)
        y = 0
        X, Y, cfoLabel = [], [], []
        label_dict = {}
        everyModCfoExampleNum = 1000
        chCfoList = [9, 10]
        modList = ['8PSK', 'BPSK', 'QAM16', 'QAM64', 'QPSK', '4ASK']
        df1 = pd.read_excel(r"..\..\data\usrp\grid\realData_2.4G_1M_CFO_0-10k\picData_1024_check.xlsx",
                            header=None)
        df2 = pd.read_excel(r"..\..\data\usrp\dft8\realData_2.4G_1M_CFO_0-10k\picData_1024_check.xlsx",
                            header=None)
        amendCFOlist1 = dict(df1.iloc[:, [0, 1]].values.tolist())
        amendCFOlist2 = dict(df2.iloc[:, [0, 1]].values.tolist())
        i = 0
        for mod in modList:
            for cfo in chCfoList:
                for i in range(everyModCfoExampleNum):
                    label_dict[y] = mod
                    data = picData[(mod, cfo, i)]
                    key = mod + '_' + str(cfo) + '_' + str(i)
                    # _, amendFreq1, _ = amendCFO.amend(data, 0, 'grid', 0.2e6,
                    #                                   freqRange=list(np.arange(-1000, 1000, 10)))
                    # if mod == '8PSK':
                    #     _, amendFreq2, _ = amendCFO.amend(data, 0, 'dft8', 0.2e6, order=8)
                    # elif mod == 'BPSK':
                    #     _, amendFreq2, _ = amendCFO.amend(data, 0, 'dft8', 0.2e6, order=2)
                    # else:
                    #     _, amendFreq2, _ = amendCFO.amend(data, 0, 'dft8', 0.2e6, order=4)
                    amendFreq1 = amendCFOlist1[key]
                    amendFreq2 = amendCFOlist2[key]
                    Y.append([[float(amendFreq1), float(amendFreq2)], y])
                    data = [np.real(data), np.imag(data)]
                    # 从 2*sampleLen 到 sampleLen*2
                    X.append(np.transpose(data))
            y += 1
        trainX, validX, testX, trainY, validY, testY, _, _, _ = train_valid_test_split(X, Y, everyModCfoExampleNum,
                                                                                       trainSplitRate=0.7,
                                                                                       validSplitRate=0.2,
                                                                                       testSplitRate=0.1)
    elif dataset == 'rml':
        model_path = r'../../models_checkpoint/CFOMendModel/origin/RML2016.10a-only-digital-modulaitons/model_weight'
        DataSavePath = '../../data/' + dataset + '/origin/RML2016.10a-only-digital-modulaitons/picData_128_RML2016.10a-only-digital-modulaitons.pkl'
        with open(DataSavePath, 'rb') as f:
            picData = pkl.load(f)
            # print(picData.keys())
        X, Y, snrLable = [], [], []
        y = 0
        modList = ['8PSK', 'BPSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK']
        chSnrList = [-2, 0, 2, 4, 6, 8]
        everyModSnrExampleNum = 125
        label_dict = {}
        df1 = pd.read_excel(r"..\..\data\rml\grid\RML2016.10a-only-digital-modulaitons\mergeData_128_check.xlsx",
                            header=None)
        df2 = pd.read_excel(r"..\..\data\rml\dft8\RML2016.10a-only-digital-modulaitons\mergeData_128_check.xlsx",
                            header=None)
        amendCFOlist1 = dict(df1.iloc[:, [0, 1]].values.tolist())
        amendCFOlist2 = dict(df2.iloc[:, [0, 1]].values.tolist())

        for mod in modList:
            for snr in chSnrList:
                for i in range(everyModSnrExampleNum):
                    label_dict[y] = mod
                    data = picData[(mod, snr, i)]
                    key = mod + '_' + str(snr) + '_' + str(i)
                    # _, amendFreq1, _ = amendCFO.amend(data, 0, 'grid', 0.2e6,
                    #                                   freqRange=list(np.arange(-1000, 1000, 10)))
                    # if mod == '8PSK':
                    #     _, amendFreq2, _ = amendCFO.amend(data, 0, 'dft8', 0.2e6, order=8)
                    # elif mod == 'BPSK':
                    #     _, amendFreq2, _ = amendCFO.amend(data, 0, 'dft8', 0.2e6, order=2)
                    # else:
                    #     _, amendFreq2, _ = amendCFO.amend(data, 0, 'dft8', 0.2e6, order=4)
                    amendFreq1 = amendCFOlist1[key]
                    amendFreq2 = amendCFOlist2[key]
                    Y.append([[float(amendFreq1), float(amendFreq2)], y])
                    data = [np.real(data), np.imag(data)]
                    # 从 2*sampleLen 到 sampleLen*2
                    X.append(np.transpose(data))
            y += 1
        trainX, validX, testX, trainY, validY, testY, _, _, _ = train_valid_test_split(X, Y, everyModSnrExampleNum,
                                                                                       trainSplitRate=0.7,
                                                                                       validSplitRate=0.2,
                                                                                       testSplitRate=0.1)

    train_empirical_label = trainY[:, 0]
    train_mod_label = trainY[:, 1]
    valid_empirical_label = validY[:, 0]
    valid_mod_label = validY[:, 1]
    test_empirical_label = testY[:, 0]
    test_mod_label = testY[:, 1]
    # a = train_empirical_label.reshape(-realData_CFO0.005, realData_CFO0.005)
    for j in range(num_epochs):
        for i in range(len(trainX) // batch_size):
            pic_trainX = trainX[i * batch_size:(i + 1) * batch_size]
            pic_train_empirical_label = train_empirical_label[i * batch_size:(i + 1) * batch_size]
            # train_loss = model.get_grad_update(pic_trainX, pic_train_empirical_label.reshape(-realData_CFO0.005, realData_CFO0.005), 0.2e6)
            train_loss = model.get_grad_update(pic_trainX, pic_train_empirical_label, 1e6)
        val_loss, min_val_loss = model.get_val_loss(validX, valid_empirical_label, 1e6, model_path)
    model = CFOMendModel(testX.shape[1:])
    model.load_weights(model_path)
    cfo_pre = model.getCFO(testX).numpy()
    cfo_pre = cfo_pre.reshape(cfo_pre.shape[0], )
    print(cfo_pre)
    fig_path = '../../results/cfo_mend_model/' + dataset + '/9-10'
    plt.rcParams['figure.figsize'] = (105 / 100, 105 / 100)
    plt.rcParams['savefig.dpi'] = 100
    for i in range(len(cfo_pre)):
        print(cfo_pre[i])
        mod = label_dict[testY[i][1]]
        path = os.path.join(fig_path, mod)
        if not os.path.exists(path):
            os.makedirs(path)
        DestinyFig(testX[i][:, 0], testX[i][:, 1], path, str(i) + 'before').plot_destiny()
        input = testX[i][:, 0] + 1j * testX[i][:, 1]
        amendData = input * np.exp(
            1j * 2 * np.pi * -cfo_pre[i] * np.linspace(0, 1 / float(1e6) * len(input), len(input)))
        DestinyFig(np.real(amendData), np.imag(amendData), path, str(i) + 'after').plot_destiny()
