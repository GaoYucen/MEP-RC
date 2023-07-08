#%% 读取模拟数据
order_list = []
f = open('data/sample_order.txt', 'rb')
for line in f.readlines():
    order = []
    can_list = line.decode().strip().split(';')
    for can in can_list[0:-1]:
        can = can.strip().split(',')
        order.append([int(x) for x in can])
    order_list.append(order)
f.close()

#%% 读取label数据
link_list = []
seq_list = []
f = open('data/sample_label.txt', 'rb')
for line in f.readlines():
    item = line.decode().strip().split(',')
    can_list = []
    can_list.append(item[2][7:])
    for i in range(3, 6):
        can_list.append(int(item[i][1:]))
    can_list.append(int(item[6][1:-1]))
    seq = []
    seq.append(int(item[7][11:]))
    for i in range(8, 11):
        seq.append(int(item[i][1:]))
    seq.append(int(item[11][1:-1]))
    link_list.append(can_list)
    seq_list.append(seq)
f.close()

#%% 读取link_feature特征数据
import pandas as pd
linkpath = "data/link_feature.txt"
link_feature = pd.read_csv(linkpath, sep='\t')

#%% 复制link_feature
link_feature_0 = link_feature.copy()
link_feature_0['link_ID'] = link_feature_0['link_ID'] * 10
link_feature_1 = link_feature.copy()
link_feature_1['link_ID'] = link_feature_1['link_ID'] * 10 + 1
link_feature = pd.concat([link_feature_0, link_feature_1])

#%% 删除无效特征
import numpy as np
# feature去掉link_ID列和geometry列
feature_used = link_feature.drop(columns=['geometry', 'Kind'])
# Drop rows with NaN values in feature_used
feature_used = feature_used.dropna(axis=1)
# 删除无效feature
feature_used = np.array(feature_used.astype('float64'))
# 对feature_used的每一列进行归一化
for i in range(feature_used.shape[1]):
    max_value = np.max(feature_used[:, i])
    feature_used[:, i] = feature_used[:, i] / max_value
# 删除含nan的列
feature_used = feature_used[:, ~np.isnan(feature_used).any(axis=0)]

# 创建打分模型
# 打分模型一般包括：线性回归模型、逻辑回归模型、支持向量机（SVM）、决策树模型、随机森林模型、梯度提升树（GBM）、深度学习模型：如多层感知机（MLP）、卷积神经网络（CNN）和循环神经网络（RNN）
# 基于Pairwise LTR实现打分评价

#%% 基于pytorch实现MLP
import torchvision
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import time


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

#%%
def read_data(filename):
    dataset = np.load(filename, allow_pickle=True)
    dataset_train = []
    for data in dataset:
        tmp_list = []
        tmp_list.append(data['label'])
        tmp_list.append(data['travel_id'])
        for _ in data['solution']:
            tmp_list.append(_)
        for _ in data['point']:
            tmp_list.append(_[0])
            tmp_list.append(_[1])
        for _ in data['square']:
            tmp_list.append(_)
        for _ in data['angle']:
            tmp_list.append(_)
        dataset_train.append(tmp_list)

    x_data = []
    y_data = []
    for i in range(len(dataset_train)):
        x_data.append(dataset_train[i][2:])
        y_data.append(int(dataset_train[i][0]))

    x_data = torch.Tensor(x_data)
    # y_data = torch.Tensor(y_data).type(torch.int64)
    y_data = torch.Tensor(y_data)

    dataset = []
    for i in range(len(x_data)):
        dataset.append([x_data[i], y_data[i]])

    return dataset


def seed_everything(seed=1234):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


#%%
tran_filepath = 'data/train_base.npy'
test_filepath = 'data/test_base.npy'

# #%%
# tran_filepath = 'data/train_norm_big.npy'
# test_filepath = 'data/test_norm_big.npy'

#%%
data_train = read_data(tran_filepath)
data_test = read_data(test_filepath)


#%% 装载数据
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)

#%%
num_i = 22  # 输入层节点数
num_h = 20  # 隐含层节点数
num_o = 1  # 输出层节点数
batch_size = 64
k = 5


class Model(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o, k):
        super(Model, self).__init__()

        self.V = torch.nn.Parameter(torch.randn(num_i, k), requires_grad=True)

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)  # S_2
        out_inter = 0.5 * (out_1 - out_2)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = x + out_inter
        x = self.sigmoid(x)

        return x

#%%
seed_everything()
model = Model(num_i, num_h, num_o, k)
# cost = torch.nn.CrossEntropyLoss()
cost = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 200
for epoch in range(epochs):
    sum_loss = 0
    train_correct = 0
    for data in data_loader_train:
        inputs, labels = data  # inputs 维度：[64,1,28,28]
        #     print(inputs.shape)
        inputs = torch.flatten(inputs, start_dim=1)  # 展平数据，转化为[64,784]
        #     print(inputs.shape)
        outputs = model(inputs)
        outputs = outputs.squeeze(-1) #转换最后一维
        optimizer.zero_grad()
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()

        # print(outputs.data)
        id = torch.tensor([1 if x > 0.5 else 0 for x in outputs.data])
        sum_loss += loss.data
        train_correct += torch.sum(id == labels.data)
    print('[%d,%d] loss:%.03f' % (epoch + 1, epochs, sum_loss / len(data_loader_train)))
    print('        correct:%.03f%%' % (100 * train_correct / len(data_train)))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

#%%
model.eval()
test_correct = 0
for data in data_loader_test:
    inputs, lables = data
    inputs, lables = Variable(inputs).cpu(), Variable(lables).cpu()
    inputs = torch.flatten(inputs, start_dim=1)  # 展并数据
    outputs = model(inputs)
    id = torch.tensor([1 if x > 0.5 else 0 for x in outputs.data])
    test_correct += torch.sum(id == lables.data)
print("correct:%.3f%%" % (100 * test_correct / len(data_test)))

#%%
data_test = read_data(test_filepath)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=False)

#%%
model.eval()
inputs_list = []
labels_list = []
acc_count = 0
for data in data_loader_test:
    inputs, lables = data
    for x in inputs:
        inputs_list.append(x)
    for x in labels:
        labels_list.append(x)
    outputs = model(inputs)
    outputs = outputs.squeeze(-1)  # 转换最后一维
    prob_list = [x for x in outputs.data]
    for i in range(int(len(outputs.data)/4)):
        score_max = max(prob_list[i * 4:(i + 1) * 4])
        index = prob_list[i * 4:(i + 1) * 4].index(score_max)
        if index == 0:
            acc_count += 1
print("correct:%.3f%%" % (100 * acc_count / (len(data_test) / 4)))



