from unlearning import *
import torch
from math import exp
import os


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


#通过acc计算子模型权重
def w(client_index, acc):
    acc_sum=float()
    for i in acc:
        acc_sum += exp(i)
    return exp(acc[client_index])/acc_sum

# 这段代码会创建一个空的融合模型，将所有子模型的参数相加，并对参数进行平均化，得到融合模型。然后，使用融合模型在测试集上进行评估，并计算准确率。
def ensemble_models():
    #少量训练集
    train_set_ensemble = ratings.sample(frac=0.1, random_state=0, axis=0)
    test_set_ensemble = ratings[~ratings.index.isin(train_set_ensemble.index)].sample(frac=0.3, random_state=0, axis=0)
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
    # 在测试集上评估融合模型
    test_data = ratings.sample(frac=0.3, random_state=0, axis=0)
    test_u_id, test_m_id, test_t_id, test_y = test_data_prep(test_data)
    test_uid = test_u_id.clone().long()
    test_mid = test_m_id.clone().long()
    test_tid = test_t_id.clone().long()
    results = []

    with torch.no_grad():
        if type(ensemble_model).__name__ == 'AFM':
            test_predictions = ensemble_model(torch.cat([test_uid, test_mid, test_tid], dim=-1))
        elif type(ensemble_model).__name__ == 'GMF' or type(ensemble_model).__name__ == 'MLP':
            test_predictions = ensemble_model(test_u_id, test_m_id, test_t_id)
        elif type(ensemble_model).__name__ == 'NCF':
            test_predictions = ensemble_model(test_u_id, test_m_id)

        test_predictions = torch.round(torch.flatten(test_predictions))
        results.append(test_predictions.detach().numpy().tolist())

    result = [item for elem in results for item in elem]
    accuracy = calculate_acc(result, test_y)
    print("Ensemble Model Accuracy: {:.4f}".format(accuracy))
    save_path = './client_ensemble.pth'
    torch.save(ensemble_model.state_dict(), save_path)





#投票法
# def pre(input_data, client_models):
#     output_n = 0
#     for model in client_models:
#         model.eval()  # 设置模型为评估模式
#         with torch.no_grad():  # 关闭梯度计算
#             input_tensor = torch.tensor(input_data)  # 将输入数据转换为张量
#             output = model(input_tensor)  # 输入模型进行预测
#             if output.item() == 1:  # 提取张量中的值进行比较
#                 output_n += 1
#     if output_n > 5:
#         print("pre:1")
#     else:
#         print("pre:0")
# def vote_fusion(client_models, test_data):# 使用投票法融合模型并测试准确率
#     epoach_count = 5  # 40
#     batchSize = 128
#     loss_value = []
#     acc_value = []
#     results=[]
#     result = []
#     test_u_id, test_m_id, test_t_id, test_y = test_data_prep(test_data)
#
#     test_uid = test_u_id.clone().long()
#     test_mid = test_m_id.clone().long()
#     test_tid = test_t_id.clone().long()
#     for i in range(10):
#         model = client_models[i]
#         model_name = type(model).__name__
#         if model_name == 'AFM':
#             test_predictions = model(torch.cat([test_uid, test_mid, test_tid], dim=-1))
#         elif model_name == 'GMF' or model_name == 'MLP':
#             test_predictions = model(test_uid, test_mid, test_tid)
#         elif model_name == 'NCF':
#             test_predictions = model(test_uid, test_mid)
#
#         test_predictions = torch.round(torch.flatten(test_predictions))
#         results.append(test_predictions.detach().numpy().tolist())
#
#         list2 = [item for elem in results for item in elem]
#         if len(result)==0:
#             result = list2
#         else:
#             list1=[]
#             for m, n in zip(result,list2):
#                 per_list = m + n
#                 list1.append(per_list)
#             result=list1
#     for i in result:
#         if i>5:
#             i=1
#         else:
#             i=0
#     acc = calculate_acc(result, test_y)
#     acc_value.append(acc)
#     print('acc:{:.4f}'.format(acc))
# vote_fusion(client_models, test_set)