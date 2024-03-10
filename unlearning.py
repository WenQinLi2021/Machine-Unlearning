import math

import pandas as pd
import ydata_profiling
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import re
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
from sklearn.utils import shuffle
import torch as F
import itertools
import copy
from math import exp
import os
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y
import sys

tqdm.pandas()


def calculate_acc(predictions, truth):  # 计算准确率
    hit = 0
    for i in range(len(predictions)):
        if predictions[i] == truth[i]:
            hit = hit + 1
    return hit / len(predictions)

def calculate_f1(y_true, y_pred):
    """
    Calculates F1 Score, Precision, and Recall
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return f1
def calculate_auc(y_true, y_pred):
    """
    Compute AUC value.

    Parameters:
    - y_true: a list of ground truth binary labels (0 or 1)
    - y_pred: a list of predicted continuous scores

    Returns:
    - AUC value
    """
    return roc_auc_score(y_true, y_pred)




ratings = pd.read_csv("C:/Users/86180/Desktop/data副本.csv", encoding='gbk')
ratings = ratings.drop(labels=['time', 'name', 'type'], axis=1)
# ratings = shuffle(ratings) #Shuffle data
# train_set = ratings.sample(frac=0.7,random_state=0,axis=0)
test_set = ratings.sample(frac=0.3, random_state=0, axis=0)


def training_data_prep(data_set):
    data = data_set
    data = shuffle(data)

    y = torch.tensor([[el] for el in data['rating']])

    u_id = torch.tensor([[el] for el in data['userId']])

    m_id = torch.tensor([[el] for el in data['movieId']])

    t_id = torch.tensor([[el] for el in data['type_id']])

    return u_id, m_id, t_id, y


def test_data_prep(data_set):
    test_y = torch.tensor([[el] for el in data_set['rating']])

    test_u_id = torch.tensor([[el] for el in data_set['userId']])

    test_m_id = torch.tensor([[el] for el in data_set['movieId']])

    test_t_id = torch.tensor([[el] for el in data_set['type_id']])

    return test_u_id, test_m_id, test_t_id, test_y


# model = GMF(82535,1302,24,64)
class GMF(nn.Module):
    def __init__(self, user_num, movie_num, type_num, factor_num):
        super(GMF, self).__init__()
        user_num = 82534 + 1
        movie_num = 1301 + 1
        type_num = 23 + 1
        factor_num = 64
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_movie_GMF = nn.Embedding(movie_num, factor_num)
        self.embed_type_GMF = nn.Embedding(type_num, factor_num)
        self.predict_layer = nn.Linear(factor_num, 1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_movie_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_type_GMF.weight, std=0.01)

    def forward(self, user, movie, type_id):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_movie_GMF = self.embed_movie_GMF(movie)
        embed_type_GMF = self.embed_type_GMF(type_id)
        output_GMF = embed_user_GMF * embed_movie_GMF * embed_type_GMF
        prediction = self.predict_layer(output_GMF)
        return prediction.view(-1, 1)


# model = MLP(82535,1302,24,256,128,16)
class MLP(nn.Module):

    def __init__(self, user_num, movie_num, type_num, user_emb_size, moive_emb_size, type_emb_size):
        super(MLP, self).__init__()
        self.user_embedding = torch.nn.Embedding(user_num, user_emb_size)
        self.movie_embedding = torch.nn.Embedding(movie_num, moive_emb_size)
        self.type_embedding = torch.nn.Embedding(type_num, type_emb_size)

        self.ReLU_activation = nn.ReLU()
        #         self.tanh_activation =  nn.Tanh()
        #         self.sigmoid_activation = nn.Sigmoid()

        self.fc1 = nn.Linear(moive_emb_size + user_emb_size + type_emb_size, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 64)  # 仿造老师多加一层

        self.fc4 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, movie_id, type_id):
        emb1 = self.user_embedding(user_id)
        # print('1:',emb1.size())
        emb2 = self.movie_embedding(movie_id)
        # print('2:',emb2.size())
        # print(type_id,type_id.size())
        emb3 = self.type_embedding(type_id)
        # print('3:',emb3.size(),type_id)

        x1 = torch.cat([emb1, emb2], 2)  # cat是将两个张量连接起来
        x = torch.cat([x1, emb3], 2)

        x = x.view(len(user_id), -1)

        h1 = self.ReLU_activation(self.fc1(x))

        h2 = self.ReLU_activation(self.fc2(h1))

        h3 = self.ReLU_activation(self.fc3(h2))

        h4 = self.fc4(h3)

        result = self.sigmoid(h4)

        return result


# model = NCF(82535, 1302, 64, 3,0.1,'NCF')，里面mlp和gmf均不带type
class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout, model, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,
                                     a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(
                self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                    self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                          self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat).view(len(concat), -1)

        prediction = self.sigmoid(prediction)
        return prediction


# model = AFM(feature_columns, mode='att', hidden_units = [128,128，64, 32], dropout=0.5, useDNN=True)/
class Dnn(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units: 列表， 每个元素表示每一层的神经单元个数， 比如[256, 128, 64], 两层网络， 第一层神经单元128， 第二层64， 第一个维度是输入维度
        dropout = 0.
        """
        super(Dnn, self).__init__()

        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        x = self.dropout(x)
        return x


class Attention_layer(nn.Module):
    def __init__(self, att_units):
        """
        :param att_units: [embed_dim, att_vector]
        """
        super(Attention_layer, self).__init__()

        self.att_w = nn.Linear(att_units[0], att_units[1])  # 8*8
        self.att_dense = nn.Linear(att_units[1], 1)  # 8*1

    def forward(self, bi_interaction):  # bi_interaction (None, (field_num*(field_num-1)_/2, embed_dim)
        a = self.att_w(bi_interaction)  # (None, (field_num*(field_num-1)_/2, t)  这里是维度变化32*325*8→ 32*325*8
        a = F.relu(a)  # (None, (field_num*(field_num-1)_/2, t)  非线性激活
        att_scores = self.att_dense(a)  # (None, (field_num*(field_num-1)_/2, 1) 再次进行维度变化 32*325*8→ 32*325*1
        att_weight = F.softmax(att_scores, dim=1)  # (None, (field_num*(field_num-1)_/2, 1)    32*325*1  对分数进行0-1范围限定

        att_out = torch.sum(att_weight * bi_interaction, dim=1)  # (None, embed_dim)     32*325*8  求和后→32*8
        return att_out


class AFM(nn.Module):
    def __init__(self, feature_columns, mode, hidden_units, att_vector=8, dropout=0.5, useDNN=False):
        """
        AFM:
        :param feature_columns: 特征信息， 这个传入的是fea_cols array[0] dense_info  array[1] sparse_info
        :param mode: A string, 三种模式, 'max': max pooling, 'avg': average pooling 'att', Attention
        :param hidden_units: DNN网络的隐藏单元个数， 一个列表的形式， 列表的长度代表层数， 每个元素代表每一层神经元个数， lambda文里面没加
        :param att_vector: 注意力网络的隐藏层单元个数
        :param dropout: Dropout比率
        :param useDNN: 默认不使用DNN网络
        """
        super(AFM, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        self.mode = mode
        self.useDNN = useDNN

        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        # 如果是注意机制的话，这里需要加一个注意力网络
        if self.mode == 'att':
            self.attention = Attention_layer([self.sparse_feature_cols[0]['embed_dim'], att_vector])

        # 如果使用DNN的话， 这里需要初始化DNN网络
        if self.useDNN:
            # 这里要注意Pytorch的linear和tf的dense的不同之处， 前者的linear需要输入特征和输出特征维度，
            # 而传入的hidden_units的第一个是第一层隐藏的神经单元个数，这里需要加个输入维度
            self.fea_num = len(self.dense_feature_cols) + self.sparse_feature_cols[0]['embed_dim']  # 13*8=21
            hidden_units.insert(0, self.fea_num)  # [21, 128, 64, 32]

            self.bn = nn.BatchNorm1d(self.fea_num)
            self.dnn_network = Dnn(hidden_units, dropout)
            self.nn_final_linear = nn.Linear(hidden_units[-1], 1)
        else:
            self.fea_num = len(self.dense_feature_cols) + self.sparse_feature_cols[0]['embed_dim']
            self.nn_final_linear = nn.Linear(self.fea_num, 1)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()  # 转成long类型才能作为nn.embedding的输入
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]
        sparse_embeds = torch.stack(sparse_embeds)  # embedding堆起来， (field_dim, None, embed_dim)   26*32*8
        sparse_embeds = sparse_embeds.permute((1, 0, 2))  # 32*26*8
        # 这里得到embedding向量之后 sparse_embeds(None, field_num, embed_dim)
        # 下面进行两两交叉， 注意这时候不能加和了，也就是NFM的那个计算公式不能用， 这里两两交叉的结果要进入Attention
        # 两两交叉enbedding之后的结果是一个(None, (field_num*field_num-1)/2, embed_dim)
        # 这里实现的时候采用一个技巧就是组合
        # 比如fild_num有4个的话，那么组合embeding就是[0,1] [0,2],[0,3],[1,2],[1,3],[2,3]位置的embedding乘积操作
        first = []
        second = []
        for f, s in itertools.combinations(range(sparse_embeds.shape[1]), 2):  # 这里就是从前面的（0-26）  产生2配对  n*（n-1）/2
            first.append(f)  # 325
            second.append(s)  # 325
        # 取出first位置的embedding  假设field是3的话，就是[0, 0, 0, 1, 1, 2]位置的embedding
        p = sparse_embeds[:, first, :]  # (None, (field_num*(field_num-1)_/2, embed_dim)
        q = sparse_embeds[:, second, :]  # (None, (field_num*(field_num-1)_/2, embed_dim)
        bi_interaction = p * q  # (None, (field_num*(field_num-1)_/2, embed_dim)  32*325*8

        if self.mode == 'max':
            att_out = torch.sum(bi_interaction, dim=1)  # (None, embed_dim)
        elif self.mode == 'avg':
            att_out = torch.mean(bi_interaction, dim=1)  # (None, embed_dim)
        else:
            # 注意力网络
            att_out = self.attention(bi_interaction)  # (None, embed_dim)  32*8

        # 把离散特征和连续特征进行拼接
        x = torch.cat([att_out, dense_inputs], dim=-1)  # 32*21

        if not self.useDNN:
            outputs = F.sigmoid(self.nn_final_linear(x))
        else:
            # BatchNormalization
            x = self.bn(x)
            # deep
            dnn_outputs = self.nn_final_linear(self.dnn_network(x))  # 32*1
            outputs = F.sigmoid(dnn_outputs)

        return outputs


# 训练模型方法
def train_model(model, optimizer, loss_fn, training_data, test_data, batchSize, epoach_count, client_id):
    if client_id is None:
        pass  # 什么也不做
    elif client_id == -1:
        print('global')
    else:
        print('Client {}'.format(client_id))
    loss_value = []
    acc_value = []
    f1_values = []
    ndcg_values = []
    test_u_id, test_m_id, test_t_id, test_y = test_data_prep(test_data)

    for epoch in range(epoach_count):
        u_id, m_id, t_id, y = training_data_prep(training_data)

        numOfMinibatches = int(len(y) / batchSize) + 1
        numOfLastMinibatch = len(y) % batchSize

        for batchID in range(numOfMinibatches):
            if batchID == numOfMinibatches - 1:
                numbOfBatches = numOfLastMinibatch
            else:
                numbOfBatches = batchSize
            leftIndex = batchID * batchSize
            rightIndex = leftIndex + numbOfBatches
            uid = u_id[leftIndex: rightIndex].clone().long()
            mid = m_id[leftIndex: rightIndex].clone().long()
            tid = t_id[leftIndex: rightIndex].clone().long()
            model_name = type(model).__name__

            if model_name == 'AFM':
                predictions = model(torch.cat([uid, mid, tid], dim=-1))
            elif model_name == 'GMF' or model_name == 'MLP':
                predictions = model(uid, mid, tid)
            elif model_name == 'NCF':
                predictions = model(uid, mid)

            loss = loss_fn(predictions, y[leftIndex: rightIndex].float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value.append(loss.item())

            # Testing
            if batchID % 500 == 0:

                test_numOfMinibatches = int(len(test_u_id) / batchSize) + 1
                test_numOfLastMinibatch = len(test_u_id) % batchSize
                final_results2 = []
                results = []

                for test_batchID in range(test_numOfMinibatches):
                    if test_batchID == test_numOfMinibatches - 1:
                        test_numbOfBatches = test_numOfLastMinibatch
                    else:
                        test_numbOfBatches = batchSize
                    test_leftIndex = test_batchID * batchSize
                    test_rightIndex = test_leftIndex + test_numbOfBatches
                    test_uid = test_u_id[test_leftIndex: test_rightIndex].clone().long()
                    test_mid = test_m_id[test_leftIndex: test_rightIndex].clone().long()
                    test_tid = test_t_id[test_leftIndex: test_rightIndex].clone().long()

                    if model_name == 'AFM':
                        test_predictions = model(torch.cat([test_uid, test_mid, test_tid], dim=-1))
                    elif model_name == 'GMF' or model_name == 'MLP':
                        test_predictions = model(test_uid, test_mid, test_tid)
                    elif model_name == 'NCF':
                        test_predictions = model(test_uid, test_mid)

                    test_predictions = torch.round(torch.flatten(test_predictions))
                    results.append(test_predictions.detach().numpy().tolist())

                result = [item for elem in results for item in elem]
                f1 = calculate_f1(test_y, result)
                auc = calculate_auc(test_y, result)
                acc = calculate_acc(result, test_y)

                acc_value.append(acc)
                f1_values.append(f1)
                print('Epoch[{}/{}],loss:{:.4f},acc:{:.4f},f1:{:.4f},auc:{:.4f}'.format(epoch, epoach_count, loss.item(), acc, f1, auc))



    return loss_value, acc_value





# 普通模型测试
# model = AFM(feature_columns, mode='att', hidden_units = [128,64, 32], dropout=0.5, useDNN=True)
# loss_value, acc_value = train_model(model, optimizer, MSELoss, train_set, test_set, batchSize, epoach_count,None)
# print(f"Client {client_id} - Loss: {loss_value}, Accuracy: {acc_value}")

user_num = 82534 + 1
movie_num = 1301 + 1
type_num = 23 + 1
user_emb_size = 128  # 256,AFM全128
movie_emb_size = 128  # 128
type_emb_size = 128  # 16
feature_columns = [
    [],  # 这里是连续特征的信息，由于没有连续特征，所以留空
    [
        {'feat_num': user_num, 'embed_dim': user_emb_size},  # 用户特征
        {'feat_num': movie_num, 'embed_dim': movie_emb_size},  # 电影特征
        {'feat_num': type_num, 'embed_dim': type_emb_size},  # 类型特征
    ]
]
#model = AFM(feature_columns, mode='att', hidden_units=[128, 128, 64, 32], dropout=0.5, useDNN=True)

#optimizer = optim.Adam(model.parameters(), lr=0.001)
MSELoss = nn.BCELoss()

counts = {}
for i in ratings['userId']:
    counts[i] = counts.get(i, 0) + 1
count = 0
max_epoach = [0]
max_epoach_userid = []
epoach_n = 0
for i in range(len(counts)):
    count += counts[i]
    if count > 90000:
        max_epoach.append(count + max_epoach[epoach_n])
        max_epoach_userid.append(i)
        epoach_n += 1
        count = 0
max_epoach[-1] = 914755
max_epoach_userid[-1] = 82534


def client_model_train():
    print("全部子模型训练开始")
    epoach_count = 2# 40
    batchSize = 128
    acc = []
    # test_u_id,test_m_id,test_t_id,test_y = test_data_prep()
    # ground_truth = ground_truth.detach().numpy().tolist()

    # 子模型训练
    # 主循环
    for client_id in range(10):
        # 创建客户端模型和优化器
        client_model = AFM(feature_columns, mode='att', hidden_units=[128, 128, 64, 32], dropout=0.5, useDNN=True)
        #client_model = MLP(user_num, movie_num, type_num, 256, 128, 16)
        optimizer = optim.Adam(client_model.parameters(), lr=0.0001)


        # 准备客户端数据集
        client_data = ratings.loc[ratings.index[max_epoach[client_id]:max_epoach[client_id + 1]]]
        client_data=shuffle(client_data)
        # print(f"Client {client_id} Data:")
        # print(client_data)
        # test_set = ratings[~ratings.index.isin(train_set.index)]
        # 训练客户端模型
        std_time = time.time()
        loss_value, acc_value = train_model(client_model, optimizer, MSELoss, client_data,
                                            test_set, batchSize, epoach_count, client_id)
        end_time = time.time()
        time_trainCM = (std_time - end_time)
        #子模型训练时间
        print("client_"+str(client_id)+"： time consuming = {} seconds".format(-time_trainCM))
        save_path = './client_'+str(client_id)+'.pth'
        torch.save(client_model.state_dict(), save_path)
        acc.append(acc_value[-1])
        # 打印客户端的损失值和准确率值
        # print(f"Client {client_id} - Loss: {loss_value}, Accuracy: {acc_value}")
    print("全部子模型训练结束")
    F = open(r'acc.txt', 'w')
    for i in acc:
        F.write(str(i) + '\n')
    F.close()


#对删除数据所在的子模型重训练
def client_model_retrain( client_index, client_models, client_data,  acc):
    print("子模型重训练开始")
    epoach_count = 2 # 40
    batchSize = 128

    client_model = AFM(feature_columns, mode='att', hidden_units=[128, 128, 64, 32], dropout=0.5, useDNN=True)
    optimizer = optim.Adam(client_model.parameters(), lr=0.0001)
    std_time = time.time()
    loss_value, acc_value = train_model(client_model, optimizer, MSELoss, client_data, test_set,
                batchSize, epoach_count, client_index)
    end_time = time.time()
    time_retrain = (std_time - end_time)
    save_path = './client_' + str(client_index) + '.pth'
    torch.save(client_models[client_index].state_dict(), save_path)
    print("已保存重训练后的client"+str(client_index))
    # 子模型训练时间
    print("client_" + str(client_index) + "： update time consuming = {} seconds".format(-time_retrain))
    acc[client_index] = acc_value[-1]
    print("子模型重训练结束")
    return  acc


#删除指定多个userId

def model_unlearn( client_models, acc):
    print("遗忘指定用户信息开始")
    ratings = pd.read_csv("data副本.csv", encoding='gbk')
    ratings = ratings.drop(labels=['time', 'name', 'type'], axis=1)
    client_id=1
    # 获取模型的保存路径
    save_path = './client_' + str(1) + '.pth'
    # 删除保存的模型文件
    os.remove(save_path)
    # 从客户端数据中删除用户信息
    client_data = ratings.loc[ratings.index[max_epoach[1]:max_epoach[1 + 1]]]
    client_data=shuffle(client_data)
    print(f"Original number of rows: {len(client_data)}")
    client_data_half = client_data.sample(frac=0.5).reset_index(drop=True)
    print(f"Number of rows after deletion: {len(client_data_half)}")
    # 在不包含用户信息的情况下重新训练客户端模型
    acc = client_model_retrain(client_id, client_models, client_data_half, acc)
    # 打印删除模型的信息
    print("遗忘指定用户信息结束")
    return   acc

#获得子模型
#client_model_train()

