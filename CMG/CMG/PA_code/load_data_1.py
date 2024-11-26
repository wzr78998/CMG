import math
from collections import defaultdict
import torch.nn as nn
import numpy as np
import os
import pickle
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA
import torch
import random
from sklearn.neighbors import KNeighborsClassifier


def get_train_test_loader_source(Data_Band_Scaler, GroundTruth,patches):
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape
    GroundTruth = GroundTruth.reshape(nRow, -1)
    [Row, Column] = np.nonzero(GroundTruth) #得到数组array中非零元素的位置（数组索引）
    GroundTruth = np.squeeze(GroundTruth.reshape(1, -1))
    # Sampling samples
    GroundTruth = GroundTruth.reshape(nRow, -1)
    da_train = {}  # Data Augmentation
    m = int(np.max(GroundTruth))  # 19
    index_all = []
    indices1 = [j for j, x in enumerate(Row.ravel().tolist()) if GroundTruth[Row[j], Column[j]] != 0]
    index_all = index_all + indices1
    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if GroundTruth[Row[j], Column[j]] == i + 1]
        if (len(indices) >= 200):
            np.random.shuffle(indices)
            da_train[i] = indices[:200]
    da_train_indices = []
    for i in range(len(da_train)):
        k = i
        da_train_indices += da_train[k]

    da_nTrain = len(da_train_indices)
    imdb = {}
    imdb['data'] = np.zeros([da_nTrain, patches, patches,nBand], dtype=np.float32)  #
    imdb['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb['set'] = np.zeros([da_nTrain], dtype=np.int64)
    feat_s = {}
    feat_s['data'] = np.zeros([int(Row.size), patches, patches, nBand], dtype=np.float32)
    feat_s['Labels'] = np.zeros([int(Row.size)], dtype=np.int64)
    RandPerm = da_train_indices
    RandPerm = np.array(RandPerm)
    index_all = np.array(index_all)
    height_S, width_S, band_S = Data_Band_Scaler.shape
    data_S = mirror_hsi(height_S, width_S, band_S, Data_Band_Scaler, patch=patches)
    for num in range(int(Row.size)):
        feat_s['data'][num, :,:, :] = data_S[Row[index_all[num]]:Row[index_all[num]]+patches,
                                 Column[index_all[num]]:Column[index_all[num]]+patches, :]
        feat_s['Labels'][num] = GroundTruth[Row[index_all[num]],
                                            Column[index_all[num]]].astype(np.int64)
    for iSample in range(da_nTrain):
        imdb['data'][iSample, :,:,:] = data_S[Row[RandPerm[iSample]]:Row[RandPerm[iSample]]+patches,
                                   Column[RandPerm[iSample]]:Column[RandPerm[iSample]]+patches, :]
        imdb['Labels'][iSample] = GroundTruth[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)
    imdb['Labels'] = imdb['Labels'] - 1
    feat_s['Labels'] = feat_s['Labels'] - 1
    print('Data is OK.')
    imdb_da_train = imdb
    return feat_s, imdb_da_train, RandPerm, Row, Column,da_train, da_train_indices
def load_source_data(path):
    with open(os.path.join(' datasets', path), 'rb') as handle:
        source_imdb = pickle.load(handle)
    print('chikusei_ok')
    source_data_train = source_imdb[0]  # (2517,2335,128)
    data = source_data_train.reshape(np.prod(source_data_train.shape[:2]), np.prod(source_data_train.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    source_data_train= data_scaler.reshape(source_data_train.shape[0], source_data_train.shape[1], source_data_train.shape[2])
    source_labels_train = source_imdb[1]  # (2517*2335)
    return source_data_train,source_labels_train
def load_data(image_file, label_file,target_data):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    if target_data=='IP':
        data_key='Indian_pines'
        label_key = label_file.split('/')[-1].split('.')[0]
    if target_data=='UP':
        data_key = 'paviaU'
        label_key = 'paviaU_gt'
    if target_data=='SA':
        data_key = 'salinas_corrected'
        label_key = 'salinas_gt'
    if target_data=='PC':
        data_key = 'pavia'
        label_key = 'pavia_gt'
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth



def applyPCAs(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[1])) ##沿光谱维度展平
    pca = PCA(n_components=numComponents, whiten=False)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],numComponents))
    return newX, pca
def get_train_data(imdb_da_train_s,feat_s,PCA_dim):
    imdb_source_data = imdb_da_train_s['data']  # 源域的元数据
    imdb_source_data,_=applyPCAs(imdb_source_data,PCA_dim)
    imdb_source_label = imdb_da_train_s['Labels']  # 源域的元数据标签
    del imdb_da_train_s
    adj=np.load(' datasets/spatial_simislarity_source12.npy')
    adj=np.array(adj,dtype=np.int8)

    del adj
    feat_s['data'], _ = applyPCAs(feat_s['data'], PCA_dim)
    feat_data_s = feat_s['data']  #
    feat_label_s = feat_s['Labels']
    del feat_s

    return  imdb_source_data,imdb_source_label,feat_data_s,feat_label_s


def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, patches):
    print(Data_Band_Scaler.shape) # (610, 340, 103)

    [nRow, nColumn, nBand] = Data_Band_Scaler.shape
    [Row, Column] = np.nonzero(GroundTruth)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(GroundTruth))  # 9
    nlabeled =shot_num_per_class

    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)
    index_all=[]

    indices1 = [j for j, x in enumerate(Row.ravel().tolist()) if GroundTruth[Row[j], Column[j]] != 0]
    index_all=index_all+indices1
    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if GroundTruth[Row[j], Column[j]] == i + 1]

        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([ nTrain + nTest,patches,patches, nBand], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices


    RandPerm = np.array(RandPerm)
    index_all ==np.array(index_all)

    height_T, width_T, band_T = Data_Band_Scaler.shape
    data_T = mirror_hsi(height_T, width_T, band_T, Data_Band_Scaler, patch=patches)
    for iSample in range(nTrain + nTest):
        imdb['data'][iSample, :,:,:] = data_T[Row[index_all[iSample]]:Row[index_all[iSample]]+patches,
                                         Column[index_all[iSample]]:Column[index_all[iSample]]+patches, :]
        imdb['Labels'][iSample] = GroundTruth[Row[index_all[iSample]], Column[index_all[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')




    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([da_nTrain,patches, patches,nBand],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][ iSample,:, :, : ] = data_T[Row[da_RandPerm[iSample]]:Row[da_RandPerm[iSample]]+patches,
            Column[da_RandPerm[iSample]]:Column[da_RandPerm[iSample]]+patches , :]
        imdb_da_train['Labels'][iSample] = GroundTruth[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return imdb, imdb_da_train ,RandPerm,Row, Column,nTrain,train,test,da_train,train_indices,test_indices,da_train_indices


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class,patches):
    imdb, imdb_da_train,RandPerm,Row, Column,nTrain ,train,test,da_train,train_indices,test_indices,da_train_indices = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class,patches=patches)  # 9 classes and 5 labeled samples per class


    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation


    target_da_datas = imdb_da_train['data']  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification


    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])


    return imdb,RandPerm,Row, Column,nTrain,train,test,da_train,train_indices,test_indices,da_train_indices



def get_samples(train_data_1,feat_data2,max_class,CLASS_NUM,SHOT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS,domain_batch):
    data1_class_choose = random.sample(range(0,max_class),CLASS_NUM)
    support_sample = np.zeros(shape=(CLASS_NUM, SHOT_NUM_PER_CLASS))
    query_sample = np.zeros(shape=(CLASS_NUM, QUERY_NUM_PER_CLASS))
    support_labels= np.zeros(shape=support_sample.shape)
    query_labels = np.zeros(shape=query_sample.shape)
    data2_sample=np.random.randint(0,feat_data2.shape[0],size=(domain_batch,))
    for i in range(CLASS_NUM):
        class_choose = data1_class_choose[i]
        random_sample_choose= random.sample(range(0, 200), SHOT_NUM_PER_CLASS+QUERY_NUM_PER_CLASS)
        random_sample_choose_suport = random_sample_choose[:SHOT_NUM_PER_CLASS]
        random_sample_choose_query = random_sample_choose[SHOT_NUM_PER_CLASS:]

        support_sample[i] = train_data_1[[class_choose], [random_sample_choose_suport]]
        support_labels[i] = i
        query_sample[i] = train_data_1[[class_choose], [random_sample_choose_query]]
        query_labels[i] = i

    support_sample= torch.LongTensor(np.squeeze(support_sample.reshape(1, -1)))
    query_sample = torch.LongTensor(np.squeeze(query_sample.reshape(1, -1)))
    data2_sample = torch.LongTensor(np.squeeze(data2_sample.reshape(1, -1))).long()
    query_labels = np.squeeze(query_labels.reshape(1, -1))
    support_labels= np.squeeze(support_labels.reshape(1, -1))
    # 打乱测试集样本顺序
    permution = np.random.permutation(np.arange(16 * 19))
    query_sample= query_sample[permution]
    query_labels = query_labels[permution]
    return support_sample,query_sample,support_labels,query_labels,data2_sample

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())




def gain_neighborhood_band(x_train, band, band_patch=3, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train = torch.Tensor(x_train.T).unsqueeze(1).detach().numpy()
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band.transpose(0,2,1)


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()

def valid_epoch(model, train_loader, valid_loader):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    features_train = np.array([])
    target_train = np.array([])
    features_test = np.array([])
    target_test = np.array([])

    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_features, batch_pred = model(batch_data)

        features_train = np.append(features_train, batch_features.data.cpu().numpy())
        target_train = np.append(target_train, batch_target.data.cpu().numpy())
    features_train = features_train.reshape(80, -1)
    KNN_classifier = KNeighborsClassifier(n_neighbors=1, p=2)
    KNN_classifier.fit(features_train, np.squeeze(target_train.reshape(1, -1)))

    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_feat, batch_pred = model(batch_data)
        pre_test = KNN_classifier.predict(batch_feat.cpu().detach().numpy())



        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, pre_test)

    return tar, pre

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

# def DataAugmentation(Data):
#     index = [range(3200)]
#     data = []
#
#     for i in len(Data):
#         data[i] = Data_T[Data[i]]
#         for j in len(data[i]):
#             data[i][j]

def merge_parameter(base_params, override_params):
    """
    Update the parameters in ``base_params`` with ``override_params``.
    Can be useful to override parsed command line arguments.

    Parameters
    ----------
    base_params : namespace or dict
        Base parameters. A key-value mapping.
    override_params : dict or None
        Parameters to override. Usually the parameters got from ``get_next_parameters()``.
        When it is none, nothing will happen.

    Returns
    -------
    namespace or dict
        The updated ``base_params``. Note that ``base_params`` will be updated inplace. The return value is
        only for convenience.
    """
    if override_params is None:
        return base_params
    is_dict = isinstance(base_params, dict)
    for k, v in override_params.items():
        if is_dict:
            if k not in base_params:
                raise ValueError('Key \'%s\' not found in base parameters.' % k)
            if type(base_params[k]) != type(v) and base_params[k] is not None:
                raise TypeError('Expected \'%s\' in override parameters to have type \'%s\', but found \'%s\'.' %
                                (k, type(base_params[k]), type(v)))
            base_params[k] = v
        else:
            if not hasattr(base_params, k):
                raise ValueError('Key \'%s\' not found in base parameters.' % k)
            if type(getattr(base_params, k)) != type(v) and getattr(base_params, k) is not None:
                raise TypeError('Expected \'%s\' in override parameters to have type \'%s\', but found \'%s\'.' %
                                (k, type(getattr(base_params, k)), type(v)))
            setattr(base_params, k, v)
    return base_params

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise
