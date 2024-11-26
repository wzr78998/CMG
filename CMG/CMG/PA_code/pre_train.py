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


from model import FE
from load_data_1 import *
from scipy.io import loadmat
import logging
from tool import *
from scipy.io import savemat
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
logger = logging.getLogger('Transformer')
def get_params():
    parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
    parser.add_argument("--seed", type=int, default=16307,help='random seed')
    parser.add_argument("--dataset_t",type = str, default = 'Pavia',help='the data of the target')
    parser.add_argument("--cls_batch_size",  type=int, default=229,help='shot num per class')
    parser.add_argument("--patches", type=int, default=7, help='patch size')
    parser.add_argument("--band_patches", type=int, default=1, help='number of related band')
    parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
    parser.add_argument('--train_begin', choices=['True', 'False'], default='True', help='mode choice')
    parser.add_argument("--lr", type=float, default=0.001, help='0.1,0.01,0.001,0.002,0.0001')
    parser.add_argument("--lr1", type=float, default=0.001, help='0.1,0.01,0.001,0.002,0.0001')
    parser.add_argument("--episode", type=int, default=400, help='episode')
    parser.add_argument('--domain_batch', type=int, default=128, help='dropout.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dropout.')
    parser.add_argument('--feature_dim', type=int, default=64, help='feature_dim.')
    parser.add_argument('--pixel_dim', type=int, default=4, help='feature_dim.')
    parser.add_argument('--sample_num_per_class', type=int, default=40, help='1,2,3,4,5.')
    parser.add_argument('--gamma', type=float, default=0.99, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--pro_num', type=int, default=142, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--n_neighbors', type=int, default=99, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--award_num', type=int, default=100, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--lamda', type=float, default=0.95, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda2', type=float, default=0.8, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--heads', type=int, default=16, help='4,8.')
    parser.add_argument('--local_kiner', type=int, default=3, help='4,8.')
    parser.add_argument('--depth', type=int, default=5, help='4,5,6,7,8.')
    parser.add_argument('--uni_dim', type=int, default=60, help='4,5,6,7,8.')
    parser.add_argument('--dropout', type=float, default=0.5, help='0,0.1.0.4,0.5,0.6.')
    parser.add_argument('--emb_dropout', type=float, default=0.5, help='0,0.1.0.4,0.5,0.6.')
    parser.add_argument('--GPU', type=int, default=0, help='0,1,2.')
    parser.add_argument('--epoch', type=int, default=100, help='0,1,2.')
    args, _  = parser.parse_known_args()
    return args
def pre_train():
    params = vars(get_params())

    ce=nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        same_seeds(params['seed'])
        if params['dataset_t'] == 'Pavia':
            data_source = sio.loadmat('../data/Pavia/paviaU.mat')['ori_data']
            GroundTruth_source = sio.loadmat('../data/Pavia/paviaU_7gt.mat')['map']
            data_target = sio.loadmat('../data/Pavia/paviaC.mat')['ori_data']
            GroundTruth_target = sio.loadmat('../data/Pavia/paviaC_7gt.mat')['map']
            h_s, w_s, b_s = data_source.shape
            h_t, w_t, b_t = data_target.shape
            from sklearn import preprocessing

            data_source = preprocessing.maxabs_scale(data_source.reshape(-1, b_s)).reshape(h_s, w_s, b_s)
            data_target = preprocessing.maxabs_scale(data_target.reshape(-1, b_t)).reshape(h_t, w_t, b_t)
            feat_data_s = get_loader(data_source, GroundTruth_source, params['patches'])
            feat_data_t = get_loader(data_target, GroundTruth_target, params['patches'])
            data_s = feat_data_s['data']
            label_s = feat_data_s['Labels'] - 1
            data_t = feat_data_t['data']
            label_t = feat_data_t['Labels'] - 1

            [Row_s, Column_s] = np.nonzero(GroundTruth_source)
            [Row_t, Column_t] = np.nonzero(GroundTruth_target)

        
        num_class = np.max(label_s) + 1



        print('data split ok')
        feture_encoder= FE(
            image_size=b_s,
            near_band=params['band_patches'],
            num_patches=params['patches'] ** 2,
            patch_size=params['patches'],
            num_classes=num_class,
            dim=params['feature_dim'],
            pixel_dim=params['pixel_dim'],
            depth=params['depth'],
            heads=params['heads'],
            mlp_dim=8,
            dropout=params['dropout'],
            emb_dropout=params['emb_dropout'],
            mode=params['mode'],
            GPU=params['GPU'],
            local_kiner=params['local_kiner']
        )



        feture_encoder.apply(weights_init).cuda()




        feture_encoder_op = torch.optim.AdamW(feture_encoder.parameters(),lr=params['lr'])


        last_accuracy=0
        all_index=[]
        pro_s=[]
        class_num_list=[]

        for i in range(num_class):
            all_index.append(np.where(label_s == i)[0])
            pro_s.append(data_s[all_index[i]])
            class_num_list.append(len(np.where(label_s == i)[0]))






    for episode in range(params['episode']):
       train_s=torch.zeros(size=(num_class,params['cls_batch_size'],params['patches'],params['patches'],b_s)).cuda()
       train_s_pro = torch.zeros(
           size=(num_class, params['pro_num'], params['patches'], params['patches'], b_s)).cuda()
       train_s_l = torch.zeros(
           size=(num_class, params['cls_batch_size'], 1)).cuda()

       for i in range(num_class):
           rand_choose=random.sample(range(0,class_num_list[i]),params['cls_batch_size'])
           rand_choose_pro = random.sample(range(0, class_num_list[i]), params['pro_num'])
           class_ind=all_index[i][rand_choose]
           class_ind_pro = all_index[i][rand_choose_pro]
           train_s[i]=torch.tensor(data_s[class_ind]).cuda()
           train_s_pro[i] = torch.tensor(data_s[class_ind_pro]).cuda()
           train_s_l[i]=i



       feture_s=feture_encoder(train_s.reshape(-1,params['patches']*params['patches'],b_s) )
       feture_s_pro = feture_encoder(train_s_pro.reshape(-1, params['patches'] * params['patches'], b_s) )

       feture_s_pro=torch.mean(feture_s_pro.reshape(num_class,params['pro_num'],feture_s_pro.shape[-1]),1)
       logits=euclidean_metric(feture_s,feture_s_pro)
       loss=ce(nn.Softmax(1)(logits),train_s_l.long().reshape(-1,))
       if episode%10==0:
           print("loss:",float(loss))
       feture_encoder.zero_grad()
       loss.backward()
       feture_encoder_op.step()
       if episode%100==0 :
         feture_encoder.eval()
         torch.save(feture_encoder.state_dict(),
                    str("./checkpoints/CMR_FE_Pre" + '_HS' + ".pkl"))

    return torch.load("./checkpoints_PA/FE_Pre.pkl")


