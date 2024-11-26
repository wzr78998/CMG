

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


from model import FE,  B_VAE_1
from load_data_1 import *
from scipy.io import loadmat

import logging
from scipy.io import savemat
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix

logger = logging.getLogger('Transformer')


import torch
import torch.nn.functional as F
def ACE(featureidx,numsaples,z_cp,feature):#计算ace向量
    bs,zdim=feature.shape #bs:200,zdim:64
    # feature_do=feature.clone()
    zdo=torch.randn(bs,zdim,requires_grad=True).cuda()   #(1,200,64)生成随机向量zdo
    # feature_do[:, ~torch.tensor([featureidx] * zdim).bool()] = zdo[:, ~torch.tensor([featureidx] * zdim).bool()]  #将第featureidx个维度替换为feature中的对应维度
    zdo[:,featureidx]=feature[:,featureidx]
    sample=euclidean_metric(zdo,z_cp) #将zdo输入分类器进行分类得到预测值（200，7）
    ACEdo=sample #ace向量即为（200，7）的预测向量

    zrand=torch.randn(bs,zdim,requires_grad=True).cuda() #随机生成向量zrand
    sample = euclidean_metric( zrand,z_cp)#直接进入分类器
    ACEbaseline=sample  #得到基线向量acebaseline
    ace=ACEbaseline-ACEdo   #最终的ace向量即为acedo-acebaseline，维度为（200，7）
    return(ace)
def contrastive(z, z_cp, labels,params):
    # 计算ace向量
    numfeature = z.shape[1]
    ace = []
    for i in range(numfeature):
        ace.append(ACE(featureidx=i, numsaples=1, z_cp=z_cp, feature=z))  # 根据类隐变量计算平均因果效应ace的值
    acematrix = torch.stack(ace, dim=1) / (torch.stack(ace, dim=1).norm(dim=1).unsqueeze(1) + 1e-8)
    acematrix = acematrix.cpu()
    class_num=int(torch.max(labels))+1
    # 对比学习三元组构建
    con_acematrix = acematrix.reshape(-1, 64 *  class_num)

    # 初始化锚点、正样本和负样本的索引列表
    anchors = []
    positives = []
    negatives = []

    # 为每个锚点选择多个正样本和负样本
    for label in range( class_num):  # 遍历所有类别
        # 找到当前类别的所有样本索引
        label_indices = (labels == label).nonzero(as_tuple=False).squeeze()

        # 随机选择一个锚点
        num_samples_in_label = label_indices.numel()
        if num_samples_in_label > 0:
            anchor_index = label_indices[random.randint(0, num_samples_in_label - 1)].item()
            anchors.append(anchor_index)

            # 选择v个正样本
            if num_samples_in_label > params['pos_num']:
                pos_indices = label_indices[random.sample(range(num_samples_in_label), params['pos_num'])]
            else:
                pos_indices = label_indices.tolist()  # 如果样本数量不足15个，则选择所有样本
            positives.extend(pos_indices)

        # 选择neg_num个负样本，确保这些样本不是锚点的同类样本
        neg_indices = (labels != label).nonzero(as_tuple=False).squeeze()
        num_samples_not_label = neg_indices.numel()
        if num_samples_not_label > 0:
            if num_samples_not_label >= params['neg_num']:
                neg_indices = random.sample(range(con_acematrix.size(0)), params['neg_num'])
            else:
                neg_indices = neg_indices.tolist()  # 如果样本数量不足20个，则选择所有样本
        negatives.extend(neg_indices)

    margin = params['margin'] # 设置 margin 超参数
    triplet_loss = 0.0

    # 遍历所有锚点
    for i, anchor in enumerate(anchors):
        # 获取当前类别的样本数量
        label = labels[anchor]
        num_samples_in_label = (labels == label).sum().item()

        # 计算每个类别中实际的正样本和负样本数量
        # 确保不会选择超出实际样本数量的索引
        pos_sample_count = min(params['pos_num'], num_samples_in_label - 1)  # 减1是为了避免选择锚点自身作为正样本
        neg_sample_count = min(params['neg_num'], len(negatives))

        # 选择正样本和负样本
        # 这里假设 positives 和 negatives 已经被初始化并填充了足够的索引
        pos_indices = positives[i * pos_sample_count:(i + 1) * pos_sample_count]
        neg_indices = negatives[i * neg_sample_count:(i + 1) * neg_sample_count]

        # 获取对应的正样本和负样本特征向量
        pos_samples = [con_acematrix[idx] for idx in pos_indices]
        neg_samples = [con_acematrix[idx] for idx in neg_indices]

        # 获取锚点的特征向量
        anchor_feature = con_acematrix[anchor]

        # 计算三元组损失
        for pos_sample in pos_samples:
            for neg_sample in neg_samples:
                # 计算锚点与正样本之间的距离
                d_pos = torch.norm(anchor_feature - pos_sample, p=2)
                # 计算锚点与负样本之间的距离
                d_neg = torch.norm(anchor_feature - neg_sample, p=2)
                # 计算三元组损失并累加
                triplet_loss += F.relu(d_pos - d_neg + margin)

    # 计算平均损失
    num_triplets = 0
    for i, anchor in enumerate(anchors):
        # 根据每个类别的实际样本数量计算三元组数量
        pos_sample_count = min(params['pos_num'], (labels == labels[anchor]).sum().item() - 1)
        neg_sample_count = min(params['neg_num'], len(negatives))
        num_triplets += pos_sample_count * neg_sample_count

    triplet_loss /= num_triplets
    return triplet_loss
from tool import *
def get_params():
    parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
    parser.add_argument("--seed", type=int, default=4822107, help='random seed')
    parser.add_argument("--TT", type=int, default=5, help='random seed')
    parser.add_argument("--dataset_t", type=str, default='HS13', help='the data of the target')
    parser.add_argument("--eps", type=float, default=0.8, help='feature_dim')
    parser.add_argument("--cls_batch_size", type=int, default=22, help='shot num per class')
    parser.add_argument("--patches", type=int, default=11, help='patch size')
    parser.add_argument("--pos_num", type=int, default=2, help='patch size')
    parser.add_argument("--neg_num", type=int, default=8, help='patch size')
    parser.add_argument("--band_patches", type=int, default=1, help='number of related band')
    parser.add_argument("--in_iter", type=int, default=5, help='number of related band')
    parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
    parser.add_argument("--lr", type=float, default=0.001, help='0.1,0.01,0.001,0.002,0.0001')
    parser.add_argument("--lr1", type=float, default=0.001, help='0.1,0.01,0.001,0.002,0.0001')
    parser.add_argument("--lr2", type=float, default=0.001, help='0.1,0.01,0.001,0.002,0.0001')
    parser.add_argument("--episode", type=int, default=100, help='episode')
    parser.add_argument('--domain_batch', type=int, default=128, help='dropout.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dropout.')
    parser.add_argument('--feature_dim', type=int, default=64, help='feature_dim.')
    parser.add_argument('--pixel_dim', type=int, default=4, help='feature_dim.')
    parser.add_argument('--sample_num_per_class', type=int, default=40, help='1,2,3,4,5.')
    parser.add_argument('--gamma', type=float, default=0.99, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--pro_num', type=int, default=231, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--n_neighbors', type=int, default=99, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--award_num', type=int, default=100, help='0.5,0.7,1.2,1.5,2.5,3.')
    parser.add_argument('--lamda1', type=float, default=0.6, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda', type=float, default=0.05, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda2', type=float, default=0.1, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda3', type=float, default=0.4, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda4', type=float, default=0.7, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda5', type=float, default=0.6, help='0.1,0.5,0.7,0.9,1.2,1.5.')
    parser.add_argument('--lamda6', type=float, default=0.2, help='0.1,0.5,0.7,0.9,1.2,1.5.')
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
    parser.add_argument('--K', type=int, default=2, help='4,8.')
    parser.add_argument('--K1', type=int, default=1, help='4,8.')
    parser.add_argument('--margin', type=float, default=0.005, help='4,8.')
    parser.add_argument('--inner_num', type=int, default=5, help='0,1,2.')
    args, _ = parser.parse_known_args()
    args, _  = parser.parse_known_args()
    return args
def run_meta(Fe_parameters,BVAE_parameters,use_parameters):
    params = vars(get_params())

    same_seeds(params['seed'])
    ce = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():


        if params['dataset_t'] == 'HS13':
            data_source = sio.loadmat('../data/Houston/Houston13.mat')['ori_data']
            GroundTruth_source = sio.loadmat('../data/Houston/Houston13_7gt.mat')['map']
            data_target = sio.loadmat('../data/Houston/Houston18.mat')['ori_data']
            GroundTruth_target = sio.loadmat('../data/Houston/Houston18_7gt.mat')['map']
            h_s, w_s, b_s = data_source.shape
            h_t, w_t, b_t = data_target.shape
            from sklearn import preprocessing
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
        feture_encoder = FE(
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
        B_vae = B_VAE_1(params['feature_dim'], params['ec_num'], params['euc_num'], params['d_num'])

        B_vae.apply(weights_init).cuda()

        B_vae_op = torch.optim.AdamW(B_vae.parameters(), lr=params['lr'])

        last_accuracy = 0
        all_index = []
        pro_s = []
        class_num_list = []

        for i in range(num_class):
            all_index.append(np.where(label_s == i)[0])
            pro_s.append(data_s[all_index[i]])
            class_num_list.append(len(np.where(label_s == i)[0]))
    feture_encoder.load_state_dict(
        Fe_parameters)
    B_vae.load_state_dict(BVAE_parameters)

    for param in feture_encoder.parameters():
        param.requires_grad = False
    for param in B_vae.decoder.parameters():
        param.requires_grad = False
    for param in B_vae.uc_encoder.parameters():
        param.requires_grad = False

    for episode in range(params['episode']):
        for inner in range(params['inner_num']):

            train_s = torch.zeros(
                size=(num_class, params['cls_batch_size'], params['patches'], params['patches'], b_s)).cuda()
            train_s_pro = torch.zeros(
                size=(num_class, params['pro_num'], params['patches'], params['patches'], b_s)).cuda()
            train_s_l = torch.zeros(
                size=(num_class, params['cls_batch_size'], 1)).cuda()

            for i in range(num_class):
                rand_choose = random.sample(range(0, class_num_list[i]), params['cls_batch_size'])
                rand_choose_pro = random.sample(range(0, class_num_list[i]), params['pro_num'])
                class_ind = all_index[i][rand_choose]
                class_ind_pro = all_index[i][rand_choose_pro]
                train_s[i] = torch.tensor(data_s[class_ind]).cuda()
                train_s_pro[i] = torch.tensor(data_s[class_ind_pro]).cuda()
                # suffer = random.sample(range(0, train_s.shape[1]), train_s.shape[1])
                train_s_l[i] = i
                # weight=nn.Sigmoid()(torch.randn(size=(train_s.shape[1],1,1,1)).cuda())
                # class_random = train_s[i][suffer]
                # train_s[i] =weight*train_s[i] +(1-weight)* class_random

            feture_s = feture_encoder(
                train_s.reshape(-1, params['patches'] * params['patches'], b_s))
            mean_c, mean_uc, var_c, var_uc, z_c, z_uc, rec_x = B_vae(feture_s, label=0)
            feture_s_pro = feture_encoder(train_s_pro.reshape(-1, params['patches'] * params['patches'], b_s))
            _, mean_ucp, _, var_ucp, z_cp1, z_ucp, _ = B_vae(feture_s_pro, label=0)
            z_cp = torch.mean(z_cp1.reshape(num_class, params['pro_num'], -1), 1)

            loss = []
            loss_mean = 0
            loss_total = 0
            z_v_list = []

            for i in range(params['K']):
                r_v = torch.randn(size=(1, 1)).cuda()

                z_v = mean_uc + r_v * torch.randn_like(mean_uc) * var_uc
                train_s_v = B_vae(torch.cat([z_c, z_v], 1), label=2)
                _, _, z_v_ = B_vae(train_s_v, label=1)
                logits = euclidean_metric(z_v_, z_cp)
                loss.append(
                    ce(nn.Softmax(1)(logits), train_s_l.long().reshape(-1, )) + params['lamda1'] * contrastive(z_v_,
                                                                                                               z_cp,
                                                                                                               train_s_l.long().reshape(
                                                                                                                   -1, ),
                                                                                                               params))
                z_v_list.append(z_v_)

            for j in range(params['K']):
                z_v_all = torch.cat(z_v_list, 0)
                loss_total = loss_total + loss[j] + params['lamda'] * ((
                        contrastive(z_v_list[j], z_cp, train_s_l.long().reshape(-1, ), params) - contrastive(
                    z_v_all, z_cp, train_s_l.long().reshape(-1, ), params))) ** 2

            del z_v_all, z_v_list
            loss_total = loss_total

            if inner == 0:
                with torch.no_grad():
                    grads_f = torch.autograd.grad(loss, B_vae.c_encoder.parameters(), create_graph=True,
                                                  allow_unused=True)
                    w_f = dict(B_vae.c_encoder.named_parameters())

            B_vae.zero_grad()

            loss_total.backward()
            B_vae_op.step()

        mean_c, mean_uc, var_c, var_uc, z_c, z_uc, rec_x = B_vae(feture_s, label=0)
        _, mean_ucp, _, var_ucp, z_cp1, z_ucp, _ = B_vae(feture_s_pro, label=0)
        z_cp = torch.mean(z_cp1.reshape(num_class, params['pro_num'], -1), 1)
        loss = []
        loss_total = 0
        for i in range(params['K1']):
            r_v = torch.randn(size=(1, 1)).cuda()
            z_v = mean_uc + r_v * torch.randn_like(mean_uc) * var_uc
            train_s_v = B_vae(torch.cat([z_c, z_v], 1), label=2)
            _, _, z_v_ = B_vae(train_s_v, label=1)
            logits = euclidean_metric(z_v_, z_cp)
            loss.append(
                ce(nn.Softmax(1)(logits), train_s_l.long().reshape(-1, )))
        for j in range(params['K1']):
            loss_total = loss_total + loss[j]

        grads_f1 = torch.autograd.grad(loss, B_vae.c_encoder.parameters(), create_graph=True,
                                       allow_unused=True)

        # 用于更新后的参数字典
        updated_w_f = {}

        # 获取 feature_encoder 的 state_dict
        state_dict = feture_encoder.state_dict()

        # 遍历模型的参数和梯度，更新仅在 state_dict 中的参数
        for (name, param), grad_f, grad_f1 in zip(w_f.items(), grads_f, grads_f1):
            if grad_f1 is None:
                updated_w_f[name] = param
            else:
                # 如果参数在 state_dict 中，则更新，否则保持原参数不变
                if name in state_dict:
                    updated_w_f[name] = param - params['lr2'] * grad_f - params['lr1'] * grad_f1
                else:
                    updated_w_f[name] = param  # 保留原始参数

        # 仅更新 state_dict 中存在的参数
        new_state_dict = {name: updated_w_f[name] for name in state_dict.keys() if name in updated_w_f}

        # 更新模型参数，只更新 state_dict 中的部分
        B_vae.c_encoder.load_state_dict(new_state_dict, strict=False)

        if episode == 0 or episode % 10 == 0:
            feture_encoder.eval()


            if use_parameters:
                feture_encoder.load_state_dict(torch.load('./checkpoints_HS/parameters/FE_Pre.pkl'))
                B_vae.load_state_dict(torch.load('./checkpoints_HS/parameters/B_VAE_finally.pkl'))



            with torch.no_grad():
                print('testing------------------------------------------')

                feture = feture_encoder(torch.tensor(data_s).cuda().reshape(-1, params['patches'] ** 2, b_s))
                _, _, z_c_s = B_vae(feture, label=1)
                KNN_classifier = KNeighborsClassifier(n_neighbors=params['n_neighbors'], p=2)
                KNN_classifier.fit(z_c_s.cpu().detach().numpy(), label_s)
                del feture
                predict = np.array([], dtype=np.int64)
                labels = np.array([], dtype=np.int64)
                bat = 100
                iter_num = data_t.shape[0] // bat
                for i in range(iter_num):
                    if i < (iter_num - 1):
                        data_t_b = data_t[i * bat:(i + 1) * bat]
                        label_t_b = label_t[i * bat:(i + 1) * bat]
                    else:
                        data_t_b = data_t[i * bat:]
                        label_t_b = label_t[i * bat:]
                    feture_t = feture_encoder(
                        torch.tensor(data_t_b).cuda().reshape(-1, params['patches'] ** 2, b_t))
                    _, _, z_c = B_vae(feture_t, label=1)
                    predict_labels = KNN_classifier.predict(z_c.cpu().detach().numpy())

                    predict = np.append(predict, predict_labels)

                    labels = np.append(labels, label_t_b)

                C = metrics.confusion_matrix(labels, predict)
                CA = np.diag(C) / np.sum(C, 1, dtype=np.float32)
                acc = 100. * len(np.where(predict == labels)[0]) / predict.shape[0]

                AA = np.mean(CA)
                kx = metrics.cohen_kappa_score(labels, predict)
                print("acc:", acc)
                print("CA", CA)
                print("AA:", AA)
                print("kappa:", kx)

                if acc > last_accuracy:
                    last_accuracy = acc
                print("best_acc:",last_accuracy)





