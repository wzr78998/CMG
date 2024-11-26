import random

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi

def get_loader(Data_Band_Scaler, GroundTruth, patches):
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape
    GroundTruth = GroundTruth.reshape(nRow, -1)
    [Row, Column] = np.nonzero(GroundTruth)  # 得到数组array中非零元素的位置（数组索引）
    GroundTruth = np.squeeze(GroundTruth.reshape(1, -1))
    # Sampling samples
    GroundTruth = GroundTruth.reshape(nRow, -1)
    da_train = {}  # Data Augmentation
    m = int(np.max(GroundTruth))  # 19
    index_all = []
    indices1 = [j for j, x in enumerate(Row.ravel().tolist()) if GroundTruth[Row[j], Column[j]] != 0]
    index_all = index_all + indices1


    feat_s = {}
    feat_s['data'] = np.zeros([int(Row.size), patches, patches, nBand], dtype=np.float32)
    feat_s['Labels'] = np.zeros([int(Row.size)], dtype=np.int64)


    index_all = np.array(index_all)
    height_S, width_S, band_S = Data_Band_Scaler.shape
    data_S = mirror_hsi(height_S, width_S, band_S, Data_Band_Scaler, patch=patches)
    for num in range(int(Row.size)):
        feat_s['data'][num, :, :, :] = data_S[Row[index_all[num]]:Row[index_all[num]] + patches,
                                       Column[index_all[num]]:Column[index_all[num]] + patches, :]
        feat_s['Labels'][num] = GroundTruth[Row[index_all[num]],
        Column[index_all[num]]].astype(np.int64)

    return feat_s

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_data(data,label):
    noz=np.where(label!=0)
    data=data[noz]
    label=label[noz]
    return data,label