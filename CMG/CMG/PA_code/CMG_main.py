import random

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
import argparse
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time

from pre_train import pre_train
from model import FE
from load_data_1 import *
from scipy.io import loadmat
import nni
import logging
from B_VAE_train import B_VAE_train
from tool import *
from scipy.io import savemat
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
logger = logging.getLogger('Transformer')
from meta import run_meta

import torch
import torch.nn.functional as F






# params = vars(get_params())



def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)


    return 0
def draw_map(best_G, seed, acc):
    hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
    for i in range(best_G.shape[0]):
        for j in range(best_G.shape[1]):
            if best_G[i][j] == 0:
                hsi_pic[i, j, :] = [0, 0, 0]
            if best_G[i][j] == 1:
                hsi_pic[i, j, :] = [0, 0, 1]
            if best_G[i][j] == 2:
                hsi_pic[i, j, :] = [0, 1, 0]
            if best_G[i][j] == 3:
                hsi_pic[i, j, :] = [0, 1, 1]
            if best_G[i][j] == 4:
                hsi_pic[i, j, :] = [1, 0, 0]
            if best_G[i][j] == 5:
                hsi_pic[i, j, :] = [1, 0, 1]
            if best_G[i][j] == 6:
                hsi_pic[i, j, :] = [1, 1, 0]
            if best_G[i][j] == 7:
                hsi_pic[i, j, :] = [0.5, 0.5, 1]
            if best_G[i][j] == 8:
                hsi_pic[i, j, :] = [0.65, 0.35, 1]
            if best_G[i][j] == 9:
                hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
            if best_G[i][j] == 10:
                hsi_pic[i, j, :] = [0.75, 1, 0.5]
            if best_G[i][j] == 11:
                hsi_pic[i, j, :] = [0.5, 1, 0.65]
            if best_G[i][j] == 12:
                hsi_pic[i, j, :] = [0.65, 0.65, 0]
            if best_G[i][j] == 13:
                hsi_pic[i, j, :] = [0.75, 1, 0.65]
            if best_G[i][j] == 14:
                hsi_pic[i, j, :] = [0, 0, 0.5]
            if best_G[i][j] == 15:
                hsi_pic[i, j, :] = [0, 1, 0.75]
            if best_G[i][j] == 16:
                hsi_pic[i, j, :] = [0.5, 0.75, 1]
  
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits
def applyPCAs(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[1])) ##沿光谱维度展平
    pca = PCA(n_components=numComponents, whiten=False)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],numComponents))
    return newX, pca
def cross_entropy(a, y):

    return (F.one_hot(y) * torch.log(a+0.000001)) / len(y)













            # 构建元训练任务














if __name__ == "__main__":
    start = time.perf_counter()
    train_begin=False
    vae_begin=False
    use_parameters=True




    try:

        print(
            "------------------------------------Pre-training-------------------------------------------------------------------")
        if train_begin:

            pre_parameters=pre_train()
        else:

            pre_parameters=torch.load('PA_code/checkpoints_PA/parameters/FE_Pre.pkl')
        print("------------------------------------BVAE-training-------------------------------------------------------------------")
        if vae_begin:

            BVAE_parameters = B_VAE_train(pre_parameters)
        else:

            BVAE_parameters = torch.load('PA_code/checkpoints_PA/parameters/B_VAE.pkl')
        print(
            "------------------------------------meta-training-------------------------------------------------------------------")
        run_meta(pre_parameters,BVAE_parameters,use_parameters)





    except Exception as exception:
        logger.exception(exception)
        raise
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
