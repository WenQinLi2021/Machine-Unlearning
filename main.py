import os
import time
from math import exp

import torch
import pandas as pd
from sklearn.utils import shuffle
from torch import optim, nn


from unlearning import client_model_train, AFM, train_model, model_unlearn, calculate_acc, test_data_prep, MSELoss, GMF, \
    NCF, MLP


"""def normal_train(model):
    ratings = pd.read_csv("data副本.csv", encoding='gbk')
    ratings = ratings.drop(labels=['time', 'name', 'type'], axis=1)
    ratings = shuffle(ratings) #Shuffle data
    train_set = ratings.sample(frac=0.7,random_state=0,axis=0)
    test_set = ratings.sample(frac=0.3, random_state=0, axis=0)

    #model =  AFM(feature_columns, mode='att', hidden_units = [128,64, 32], dropout=0.5, useDNN=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    MSELoss = nn.BCELoss()
    epoach_count =5 #40
    batchSize = 128
    loss,acc=train_model(model, optimizer, MSELoss, train_set, test_set, batchSize, epoach_count,None)"""
def w(client_index, acc):
    acc_sum=float()
    for i in acc:
        acc_sum += exp(i)
    return exp(acc[client_index])/acc_sum
def ensemble_models():
    #少量训练集
    ratings = pd.read_csv("data副本.csv", encoding='gbk')
    ratings = ratings.drop(labels=['time', 'name', 'type'], axis=1)
    ratings = shuffle(ratings)  # Shuffle data
    train_set_ensemble = ratings.sample(frac=0.1, random_state=0, axis=0)
    test_set_ensemble = ratings.sample(frac=0.3, random_state=0, axis=0)
    #读取acc文件为了按权重融合参数
    file = open('acc.txt','r')
    data = file.read().splitlines()
    acc = [float(x) for x in data]
    file.close()
    # 创建一个空的融合模型
    ensemble_model = AFM(feature_columns, mode='att', hidden_units=[128, 128, 64, 32], dropout=0.5, useDNN=True)
    weights_path = './client_0.pth'
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    ensemble_model.load_state_dict(torch.load(weights_path))

    for param_ensemble in ensemble_model.parameters():
        param_ensemble.data *= w(0, acc)
    # 融合子模型的参数
    for client_index in range(1,10):
        model = AFM(feature_columns, mode='att', hidden_units=[128, 128, 64, 32], dropout=0.5, useDNN=True)
        weights_path = './client_'+str(client_index)+'.pth'
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path))
        for param_ensemble, param_model in zip(ensemble_model.parameters(), model.parameters()):
            param_ensemble.data = param_ensemble.data + param_model.data * w(client_index, acc)

    #模型训练
    optimizers_ensemble = optim.Adam(ensemble_model.parameters(), lr=0.0001)
    loss_value_ensemble, acc_value_ensemble = train_model(ensemble_model, optimizers_ensemble, MSELoss,
                                                          train_set_ensemble, test_set_ensemble, 128, 3, -1)
    save_path = './client_ensemble.pth'
    torch.save(ensemble_model.state_dict(), save_path)

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
model =  NCF(82535, 1302, 64, 3,0.1,'NCF')
#完整数据集训练时间
std_time = time.time()
#normal_train(model)
end_time = time.time()
time_normaltrain = (std_time - end_time)
print("正常完整数据集" +"： time consuming = {} seconds".format(-time_normaltrain))

#各个子模型训练时间
client_model_train()

#得到融合模型
std_time = time.time()
ensemble_models()
end_time = time.time()
time_ensemble_models_train = (std_time - end_time)
print("融合模型" +"： time consuming = {} seconds".format(-time_ensemble_models_train))

#遗忘指定用户id
client_models = []
for client_id in range(10):
    # 读取子模型
    weight_path = './client_' + str(client_id) + '.pth'
    client_model = AFM(feature_columns, mode='att', hidden_units=[128, 128, 64, 32], dropout=0.5, useDNN=True)
    client_model.load_state_dict(torch.load(weight_path))
    # 将子模型添加到列表中
    client_models.append(client_model)
acc = []
with open('acc.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        acc.append(float(line.strip()))
acc=model_unlearn( client_models, acc)
with open('acc.txt', 'w') as f:
    for item in acc:
        f.write(str(item) + '\n')

#得到遗忘用户数据后的融合模型
std_time = time.time()
ensemble_models()
end_time = time.time()
time_ensemble_models_retrain = (std_time - end_time)
print("融合模型" +"： time consuming = {} seconds".format(-time_ensemble_models_retrain))
