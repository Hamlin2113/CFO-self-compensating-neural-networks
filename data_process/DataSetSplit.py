# -*- coding: utf-8 -*-
# @Time    : 2022/12/4 22:04
# @Author  : zwl
import random
import numpy as np


def train_valid_test_split(X: list, Y: list, everyLayerNum: int, trainSplitRate=0.7, validSplitRate=0.2,

    if type(X) is list:
        X = np.array(X)
    if type(Y) is list:
        Y = np.array(Y)
    random.seed(10)
    exampleNum = X.shape[0]
    trainIndex = []
    validIndex = []

    for layer in range(int(exampleNum / everyLayerNum)):
        trainIndex.extend(np.random.choice(range(layer * everyLayerNum, (layer + 1) * everyLayerNum),
                                           size=int(everyLayerNum * trainSplitRate), replace=False))

    leftIndex = np.array([i for i in range(exampleNum) if i not in set(trainIndex)])

    for layer in range(int(exampleNum / everyLayerNum)):
        validIndex.extend(leftIndex[np.random.choice(
            range(int(layer * everyLayerNum * (1 - trainSplitRate)),
                  int((layer + 1) * everyLayerNum * (1 - trainSplitRate))),
            size=int(everyLayerNum * validSplitRate), replace=False)])

    testIndex = list(set(leftIndex) - set(validIndex))

    np.random.shuffle(trainIndex)
    np.random.shuffle(validIndex)
    np.random.shuffle(testIndex)
    trainX = X[trainIndex]
    validX = X[validIndex]
    testX = X[testIndex]
    trainY = Y[trainIndex]
    validY = Y[validIndex]
    testY = Y[testIndex]

    return trainX, validX, testX, trainY, validY, testY, trainIndex, validIndex, testIndex
