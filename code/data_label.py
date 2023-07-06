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

#%% 构造图
import pandas as pd

mappath = '../data/bj_link_info_add_geo_2022110912_ds'
map = pd.read_csv(mappath, sep=' ', header=None, names = ['link_ID', 'SnodeID', 'EnodeID', 'fc', 'Length', 'upper_link', 'low_link'])

import networkx as nx

# 构造图结构
def graph_con(link_map):
    G = nx.DiGraph()

    for i in range(len(link_map)):
        G.add_edge(int(link_map['SnodeID'][i]), int(link_map['EnodeID'][i]), id=int(link_map['link_ID'][i]), weight=float(link_map['Length'][i]))

    return G

# 构造图
G = graph_con(map)

#%%
print(map.head(5))

#%% 定义edge_shortest_path，返回长度
# 边最短路算法（不含起止边）
def edge_shortest_path(s_link_id, e_link_id, map, graph):
    source_node = map[map['link_ID'] == s_link_id].iloc[0]['EnodeID']
    target_node = map[map['link_ID'] == e_link_id].iloc[0]['SnodeID']
    # node_path=nx.dijkstra_path(graph, source_node, target_node, weight='weight')
    length = nx.dijkstra_path_length(graph, source_node, target_node, weight='weight')
    # link_path = []
    # for i in range(len(node_path)-1):
    #     link_path.append(graph[int(node_path[i])][int(node_path[i+1])]['id'])
    # #含起止边用如下代码
    # link_path.append(s_link_id)
    # for i in len(node_path)-1:
    #     link_path.append(graph[int(node_path[i])][int(node_path[i+1])]['id'])
    # link_path.append(e_link_id)
    # length = length+link_fc_dict[s_link_id][3]+link_fc_dict[e_link_id][3]

    # return link_path, length
    return length

# 多候选link双拼路径规划
def can_carpool_route(order, map, graph):
    route_length = 0

    if (len(order) == 5):
        waypoint_list = [-1, 0, 2, 1, 3]
        # 构建candidate dict存储可能的length
        candidate_length_dict = {'-10': [], '-12': [], '02': [], '01': [], '03': [], '20': [], '21': [], '23': [],
                                 '13': [],
                                 '31': []}
        # 计算length存入candidate dict
        candidate_link_list = order
        candidate_link_list[0] = [candidate_link_list[0][0]]

        # 记录各点的多候选link的个数
        candidate_link_list_len = [len(x) for x in candidate_link_list]

        # 调用边最短路算法并存储入dict
        for key in candidate_length_dict.keys():
            for s_link_id in candidate_link_list[waypoint_list.index(int(key[0:-1]))]:
                for e_link_id in candidate_link_list[waypoint_list.index(int(key[-1]))]:
                    candidate_length_dict[key].append(
                        edge_shortest_path(s_link_id, e_link_id, map, graph))

        # 根据4条可选路径计算最短route，输出长度及route类型
        # candidate_route = [['01', '12', '23', '34'], ['01', '12', '24', '43'], ['02', '21', '13', '34'],
        #                    ['02', '21', '14', '43']]
        candidate_route = [[-1, 0, 2, 1, 3], [-1, 0, 2, 3, 1], [-1, 2, 0, 1, 3], [-1, 2, 0, 3, 1]]
        link_list = [[], [], [], []]
        candidate_route_length = [0, 0, 0, 0]

        for route_type in range(0, len(candidate_route)):
            length_min = float('inf')
            for link_0 in range(0, len(candidate_link_list[0])):  # 直接筛选出对各类型路径来说最好的长度，记录选用的link_id
                for link_1 in range(0, len(candidate_link_list[1])):
                    for link_2 in range(0, len(candidate_link_list[2])):
                        for link_3 in range(0, len(candidate_link_list[3])):
                            for link_4 in range(0, len(candidate_link_list[4])):
                                link_list_temp = [link_0, link_1, link_2, link_3, link_4]
                                length = 0
                                for i in range(0, 4):
                                    length = length + candidate_length_dict[
                                        str(candidate_route[route_type][i]) + str(candidate_route[route_type][i + 1])][
                                        link_list_temp[waypoint_list.index(candidate_route[route_type][i])] *
                                        candidate_link_list_len[
                                            waypoint_list.index(candidate_route[route_type][i + 1])] + link_list_temp[
                                            waypoint_list.index(
                                                candidate_route[route_type][i + 1])]]  # 根据选取的link确定长度所在的index
                                # 创建有效性列表
                                valid_list = [1] * len(link_list_temp)
                                for i in range(0, len(link_list_temp)-1):
                                    if candidate_link_list[i][link_list_temp[i]] == candidate_link_list[i+1][link_list_temp[i+1]]:
                                        valid_list[i+1] = 0
                                for i in range(0, len(link_list_temp)):
                                    if valid_list[i] == 1:
                                        length = length + map[map['link_ID'] == candidate_link_list[i][link_list_temp[i]]].iloc[0]['Length']  # 输出路径总长
                                if length < length_min:
                                    length_min = length
                                    candidate_route_length[route_type] = length
                                    link_list[route_type] = [candidate_link_list[x][link_list_temp[x]] for x in
                                                             range(0, len(link_list_temp))]

        # 确认最短路
        route_length = min(candidate_route_length)

        # 根据最短路确定route_sequence
        route_type = candidate_route_length.index(min(candidate_route_length))
        route_type_list = candidate_route[route_type]

        # # 打印接送驾顺序
        # print('route_length:', route_length)
        # print('route_sequence:', route_type_list)

    return route_length, link_list[route_type], route_type_list

print(can_carpool_route(order_list[0], map, G))

#%%
import pickle
res_dict = {}
for order in order_list:
    res = can_carpool_route(order, map, G)
    res_dict.update({'Length': res[0], 'Link': res[1], 'Sequence': res[2]})

with open('data/sample_label.pkl', 'wb') as tf:
    pickle.dump(res_dict, tf)

# #%% 读取
# with open('data/sample_label.pkl', 'rb') as tf:
#     res_dict = pickle.load(tf)




