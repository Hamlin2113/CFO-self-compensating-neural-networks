# -*- coding: utf-8 -*-
# @Time    : 2022/12/realData_CFO0.005 17:31
# @Author  : zwl

import numpy as np
from numpy import ndarray
from scipy import signal
from vmdpy import VMD
from PyEMD.EMD import EMD
from tools.data_process.AxisTransform import iq_to_polar, polar_to_iq


class AmendCFO():
    def __init__(self, gridNum: int, threshold: float):
        
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold shoud be within[0,1]")
        self.gridNum = gridNum
        self.threshold = threshold
        self.dataTypeDic = {0: "iq", 1: "por"}
        self.amendTypeDic = {0: "grid", 1: "dft8", 2: "cor", 3: "emd", 4: "vmd"}

    def amend(self, input: ndarray, dataType: int, amendType: str, sampleRate: float, order: int = 8,
              freqRange: list = None,fine_center=None) -> (ndarray, float, list):
        
        global amendData
        if dataType == 1:
            input = polar_to_iq(input)
        amendFreq = 0
 
        ind = [self.indic(self.count_sum_in_grid(iq_to_polar(input))), 1]
        sampleLen = len(input)
        if amendType == "grid":
            for i in range(len(freqRange)):
                tempData = input * np.exp(
                    1j * 2 * np.pi * -freqRange[i] * np.linspace(0, 1 / float(sampleRate) * sampleLen, sampleLen))
                curInd = self.indic(self.count_sum_in_grid(iq_to_polar(tempData)))
                if ind[1] > curInd:
 
                    ind[1], amendFreq = curInd, freqRange[i]
       
            amendData = input * np.exp(
                1j * 2 * np.pi * -amendFreq * np.linspace(0, 1 / float(sampleRate) * sampleLen, sampleLen))
        elif amendType == "dft8":
            fft = np.fft.fft(np.power(input, order), n=sampleLen)
            amp = abs(fft) / len(fft)
            fre = np.fft.fftfreq(d=1 / sampleRate, n=sampleLen)
            amendFreq = fre[np.argmax(amp)] / order
            amendData = input * np.exp(
                1j * 2 * np.pi * -amendFreq * np.linspace(0, 1 / float(sampleRate) * sampleLen, sampleLen))
            ind[1] = self.indic(self.count_sum_in_grid(iq_to_polar(amendData)))
        elif amendType == "cor":
            # 归一化
            input = input / np.abs(input)
            fmax = freqRange[-1]
            L = round(sampleRate / fmax) - 1
            if L >= sampleLen:
                print('The autocorrelation correction L-parameter should be less than the number of samples')
            correlation = 0
            for index in range(1, L + 1):
                cor = 0
                for i in range(index, sampleLen):
                    cor += input[i] * input[i - index].conjugate()
                correlation += cor / (sampleLen - index)
            arg = np.real(iq_to_polar(np.array([correlation])))
            amendFreq = arg / np.pi / (L + 1) * sampleRate
            if amendFreq.ndim >= 1:
                amendFreq = amendFreq[0]
            amendData = input * np.exp(
                1j * 2 * np.pi * -amendFreq * np.linspace(0, 1 / float(sampleRate) * sampleLen, sampleLen))
            ind[1] = self.indic(self.count_sum_in_grid(iq_to_polar(amendData)))
        elif amendType == "emd":
       
            emd = EMD()
            emd.emd(np.abs(input))
            print("emd1")
            imfs, res = emd.get_imfs_and_residue()
            analyticSignal = signal.hilbert(imfs)
            instantaneous_phase = np.unwrap(np.angle(analyticSignal))
            freqRange = (np.diff(instantaneous_phase) / (2.0 * np.pi) * sampleLen)
            amendFreq = np.max(freqRange[0, :])
            amendData = input * np.exp(
                1j * 2 * np.pi * -amendFreq * np.linspace(0, 1 / float(sampleRate) * sampleLen, sampleLen))
            ind[1] = self.indic(self.count_sum_in_grid(iq_to_polar(amendData)))
        elif amendType == "vmd":
            alpha = 2000  # moderate bandwidth constraint
            tau = 0.5  # noise-tolerance (no strict fidelity enforcement)
            K = 3  # 3 modes
            DC = 0  # no DC part imposed
            init = 1  # initialize omegas uniformly
            tol = 1e-7
           
            imfs, u_hat, omega = VMD(np.abs(input), alpha, tau, K, DC, init, tol)
            analyticSignal = signal.hilbert(imfs)
            instantaneous_phase = np.unwrap(np.angle(analyticSignal))
            freqRange = (np.diff(instantaneous_phase) / (2.0 * np.pi) * sampleLen)
            amendFreq = np.mean(np.sum(freqRange, 0))
            amendData = input * np.exp(
                1j * 2 * np.pi * -amendFreq * np.linspace(0, 1 / float(sampleRate) * sampleLen, sampleLen))
            ind[1] = self.indic(self.count_sum_in_grid(iq_to_polar(amendData)))
        return amendData, amendFreq, ind

    def count_sum_in_grid(self, polarData: ndarray) -> ndarray:
       
        grid = np.linspace(-np.pi - 0.5, np.pi + 0.5, self.gridNum)
        array1 = np.sort_complex(polarData)
        record = np.zeros(self.gridNum - 1)
        j = 0
        for i in range(self.gridNum - 1):
            if j == len(polarData):
                break
            while j != len(polarData) and array1[j] <= grid[i + 1]:
                record[i] += 1
                j += 1
        return record

    def indic(self, record: ndarray) -> float:
     
        rec = np.sort(record)
        return 1 - sum(rec[-int((self.gridNum - 1) * self.threshold):]) / sum(rec)


if __name__ == '__main__':
    a = AmendCFO(31, 0.25)
