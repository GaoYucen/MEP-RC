#%%
# Import necessary libraries
import os
import pandas as pd
import random

# Define the file path
filepath = "../data/link_feature.txt"
mappath = '../data/bj_link_info_add_geo_2022110912_ds'

link_map = pd.read_csv(filepath, sep='\t')
map = pd.read_csv(mappath, sep=' ', header=None, names = ['link_ID', 'SnodeID', 'EnodeID', 'fc', 'Length', 'upper_link', 'low_link'])

#%%
link_list = link_map['link_ID'].tolist()
map_list = map['link_ID'].tolist()

#%% 删除map中link_ID重复的项→会导致构图错误
# map = map.drop_duplicates(subset=['link_ID'], keep='last')
# print(len(map))

#%% 由Snode、enode构造networkx的graph
import networkx as nx

# 构造图结构
def graph_con(link_map):
    G = nx.DiGraph()

    for i in range(len(link_map)):
        G.add_edge(int(link_map['SnodeID'][i]), int(link_map['EnodeID'][i]), id=int(link_map['link_ID'][i]), weight=float(link_map['Length'][i]))

        # if map['link_ID'][i] * 10 + 0 in link_list:
        #     G.add_edge(int(map['SnodeID'][i]), int(map['EnodeID'][i]), id=int(map['link_ID'][i]),
        #                weight=float(map['Length'][i]))
        # if map['link_ID'][i] * 10 + 1 in link_list:
        #     G.add_edge(int(map['EnodeID'][i]), int(map['SnodeID'][i]), id=int(map['link_ID'][i]),
        #                weight=float(map['Length'][i]))

    return G

# 生成一个空的无向图
G = graph_con(map)

# 统计边和点的数目
# print(G.number_of_nodes())
# print(G.number_of_edges())

# # 找出重复项
# duplicate = [n for n in link_ID if link_ID.count(n) > 1]

#%% 生成随机订单
order_num = 10000
# 生成order_num个样本
# 双拼
# 从link_ID中挑选边
def link_id_determine(link_list, map_list):
    flag = 0
    while flag == 0:
        link_ID = link_list[int(random.random() * len(link_list))]
        if random.random() < 0.5:
            if link_ID * 10 + 0 in map_list:
                link_ID = link_ID * 10 + 0
                flag = 1
            elif link_ID * 10 + 1 in map_list:
                link_ID = link_ID * 10 + 1
                flag = 1
        else:
            if link_ID * 10 + 1 in map_list:
                link_ID = link_ID * 10 + 1
                flag = 1
            elif link_ID * 10 + 0 in map_list:
                link_ID = link_ID * 10 + 0
                flag = 1

    return link_ID

order_list = []
for i in range(order_num):
    print(i)
    order = []
    link_ID = link_id_determine(link_list, map_list)
    order.append([link_ID])
    for j in range(1, 5):
        link_ID = link_id_determine(link_list, map_list)
        # 查找edge的前续边，后续边
        snode = map[map['link_ID']==link_ID].iloc[0]['SnodeID']
        pred = list(G.predecessors(snode))
        enode = map[map['link_ID']==link_ID].iloc[0]['EnodeID']
        succ = list(G.successors(enode))
        can_list = []
        for node in pred:
            can_list.append(G.edges[node, snode]['id'])
        for node in succ:
            can_list.append(G.edges[enode, node]['id'])
        order.append(can_list)
        print(order)
    order_list.append(order)

#%%
f = open('../data/sample_order.txt', 'w')
for order in order_list:
    for edges in order:
        for i, edge in enumerate(edges):
            if i != len(edges)-1:
                print(edge, end=',', file=f)
            else:
                print(edge, end=';', file=f)
    print(end='\n', file=f)
f.close()





