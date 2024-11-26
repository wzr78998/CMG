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


from model import FE,B_VAE_1
from load_data_1 import *
from scipy.io import loadmat
import nni
import logging
from scipy.io import savemat
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix



logger = logging.getLogger('Transformer')

def get_params():
    parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
    parser.add_argument("--seed", type=int, default=37406,help='random seed')
    parser.add_argument("--TT", type=int, default=5, help='random seed')
    parser.add_argument("--dataset_t",type = str, default = 'HS13',help='the data of the target')
    parser.add_argument("--eps",type = float, default =0.8,help='feature_dim')
    parser.add_argument("--cls_batch_size",  type=int, default=90,help='shot num per class')
    parser.add_argument("--patches", type=int, default=11, help='patch size')
    parser.add_argument("--band_patches", type=int, default=1, help='number of related band')
    parser.add_argument("--in_iter", type=int, default=5, help='number of related band')
    parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
    parser.add_argument("--lr", type=float, default=0.001, help='0.1,0.01,0.001,0.002,0.0001')
    parser.add_argument("--lr1", type=float, default=0.00001, help='0.1,0.01,0.001,0.002,0.0001')
    parser.add_argument("--lr2", type=float, default=0.00001, help='0.1,0.01,0.001,0.002,0.0001')
    parser.add_argument("--episode", type=int, default=700, help='episode')
    parser.add_argument('--domain_batch', type=int, default=128, help='dropout.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dropout.')
    parser.add_argument('--feature_dim', type=int, default=64, help='feature_dim.')
    parser.add_argument('--pixel_dim', type=int, default=4, help='feature_dim.')
    parser.add_argument('--sample_num_per_class', type=int, default=40, help='1,2,3,4,5.')
    parser.add_argument('--gamma', type=float, default=0.99, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--pro_num', type=int, default=143, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--n_neighbors', type=int, default=99, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--award_num', type=int, default=100, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--lamda1', type=float, default=0.8, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda2', type=float, default=0.2, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda3', type=float, default=0.4, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda4', type=float, default=0.7, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda5', type=float, default=0.6, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda6', type=float, default=0.2, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda7', type=float, default=0.1, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--heads', type=int, default=16, help='4,8.')
    parser.add_argument('--ec_num', type=int, default=6, help='4,8.')
    parser.add_argument('--euc_num', type=int, default=5, help='4,8.')
    parser.add_argument('--d_num', type=int, default=2, help='4,8.')
    parser.add_argument('--local_kiner', type=int, default=3, help='4,8.')
    parser.add_argument('--depth', type=int, default=5, help='4,5,6,7,8.')
    parser.add_argument('--uni_dim', type=int, default=60, help='4,5,6,7,8.')
    parser.add_argument('--dropout', type=float, default=0.5, help='0,0.1.0.4,0.5,0.6.')
    parser.add_argument('--emb_dropout', type=float, default=0.5, help='0,0.1.0.4,0.5,0.6.')
    parser.add_argument('--GPU', type=int, default=0, help='0,1,2.')
    parser.add_argument('--epoch', type=int, default=100, help='0,1,2.')
    args, _  = parser.parse_known_args()
    return args
import torch
import torch.nn.functional as F
def KL(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
from tool import *
def B_VAE_train(paremeters):
    params = vars(get_params())
    ce=nn.CrossEntropyLoss().cuda()


    with torch.no_grad():
        same_seeds(params['seed'])

        if params['dataset_t'] == 'HS13':
            data_source= sio.loadmat('../data/Houston/Houston13.mat')['ori_data']
            GroundTruth_source = sio.loadmat('../data/Houston/Houston13_7gt.mat')['map']
            data_target= sio.loadmat('../data/Houston/Houston18.mat')['ori_data']
            GroundTruth_target = sio.loadmat('../data/Houston/Houston18_7gt.mat')['map']
            h_s, w_s, b_s = data_source.shape
            h_t, w_t, b_t = data_target.shape
            from sklearn import preprocessing
            feat_data_s = get_loader(data_source, GroundTruth_source , params['patches'])
            feat_data_t = get_loader(data_target, GroundTruth_target, params['patches'])
            data_s= feat_data_s['data']
            label_s =feat_data_s['Labels'] - 1
            data_t= feat_data_t['data']
            label_t= feat_data_t['Labels'] - 1
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
        ).cuda()
        B_vae =B_VAE_1(params['feature_dim'],params['ec_num'],params['euc_num'],params['d_num'])
        domain_decoder=B_VAE_1(params['feature_dim'],params['ec_num'],params['euc_num'],params['d_num']).decoder
        B_vae.apply(weights_init).cuda()
        domain_decoder.apply(weights_init).cuda()
        B_vae_op = torch.optim.AdamW(B_vae.parameters(), lr=params['lr'])
        domain_decoder_op= torch.optim.AdamW(domain_decoder.parameters(), lr=params['lr2'])


        last_accuracy=0
        all_index=[]
        pro_s=[]
        class_num_list=[]

        for i in range(num_class):
            all_index.append(np.where(label_s == i)[0])
            pro_s.append(data_s[all_index[i]])
            class_num_list.append(len(np.where(label_s == i)[0]))
    feture_encoder.load_state_dict(
        paremeters)
    for param in feture_encoder.parameters():
        param.requires_grad = False





    for episode in range(params['episode']):

       train_s=torch.zeros(size=(num_class,params['cls_batch_size'],params['patches'],params['patches'],b_s)).cuda()
       train_s_pro = torch.zeros(
           size=(num_class, params['pro_num'], params['patches'], params['patches'], b_s)).cuda()
       train_s_l = torch.zeros(
           size=(num_class, params['cls_batch_size'], 1)).cuda()
       train_s_l_u = torch.zeros(
           size=(num_class, params['cls_batch_size'], num_class)).cuda()

       for i in range(num_class):
           rand_choose=random.sample(range(0,class_num_list[i]),params['cls_batch_size'])
           rand_choose_pro = random.sample(range(0, class_num_list[i]), params['pro_num'])
           class_ind=all_index[i][rand_choose]
           class_ind_pro = all_index[i][rand_choose_pro]
           train_s[i]=torch.tensor(data_s[class_ind]).cuda()
           train_s_pro[i] = torch.tensor(data_s[class_ind_pro]).cuda()
           train_s_l[i]=i
           # train_s_l_u[i,:,i] = 0



       feture_s=feture_encoder(train_s.reshape(-1,params['patches']*params['patches'],b_s))

       mean_c, mean_uc, var_c, var_uc, z_c, z_uc, rec_x=B_vae(feture_s,label=0)
       feture_s_pro= feture_encoder(train_s_pro.reshape(-1, params['patches'] * params['patches'], b_s))
       _,_,z_cp= B_vae(feture_s_pro, label=1)

       z_cp=torch.mean(z_cp.reshape(num_class,params['pro_num'],-1),1)


       logits=euclidean_metric(z_c,z_cp)
       logits1 = euclidean_metric(z_uc, z_cp)
       feture_us=domain_decoder.forward(torch.cat([z_uc,z_uc],1))
       c_kl_loss=0
       for i in range(num_class):
           c_kl_loss=c_kl_loss+nn.KLDivLoss()(nn.Softmax()(torch.mean(z_c,0)),nn.Softmax()(z_cp[i]))




       kl_loss=params['lamda7']*c_kl_loss+params['lamda4']*KL(mean_c,torch.exp(var_c))+params['lamda5']*KL(mean_uc,torch.exp(var_uc))+params['lamda6']*nn.L1Loss()(nn.Softmax(1)(logits1),torch.ones(logits1.shape[0],num_class).cuda())
       loss=params['lamda1']*nn.MSELoss()(rec_x,feture_s)+params['lamda2']*ce(nn.Softmax(1)(logits),train_s_l.long().reshape(-1,))+kl_loss+params['lamda3']*nn.MSELoss()(feture_us,feture_s)
       B_vae.zero_grad()
       domain_decoder.zero_grad()
       loss.backward()
       domain_decoder_op.step()
       B_vae_op.step()
       if episode == 0 or episode % 100 == 0:

           feture_encoder.eval()



           torch.save(B_vae.state_dict(),
                              str("../HS/checkpoints_HS/B_VAE.pkl"))

    return torch.load("../HS/checkpoints_HS/B_VAE.pkl")



