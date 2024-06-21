# -*- coding: utf-8 -*-
# @Time    : 2022/12/realData_CFO0.005 16:33
# @Author  : zwl
import numpy as np
from numpy import ndarray


def iq_to_polar(input: ndarray) -> (ndarray, ndarray):

    data_I = np.real(input)
    data_Q = np.imag(input)
    # 极坐标的r与theta
    r = np.sqrt(data_I ** 2 + data_Q ** 2)
    theta = np.zeros(len(data_I))
    for i in range(len(input)):
        if data_I[i] > 0:
            theta[i] = np.arctan(data_Q[i] / data_I[i])
        elif (data_I[i] < 0) and (data_Q[i] >= 0):
            theta[i] = np.arctan(data_Q[i] / data_I[i]) + np.pi
        elif (data_I[i] < 0) and (data_Q[i] < 0):
            theta[i] = np.arctan(data_Q[i] / data_I[i]) - np.pi
        elif (data_I[i] == 0) and (data_Q[i] > 0):
            theta[i] = np.pi / 2
        elif (data_I[i] == 0) and (data_Q[i] < 0):
            theta[i] = -np.pi / 2
        else:
            theta[i] = 0
    return theta+1j*r


def polar_to_iq(input: ndarray) -> ndarray:

    return np.imag(input) * np.cos(np.real(input)) + 1j * np.imag(input) * np.sin(np.real(input))
