# MEP-RC

Environment:
- python 3.9

Structure:
- code
  - data_con.py: 根据北京路网数据以uniform random的方式选择多侯选边，形成模拟订单
  
  - data_label.py: 针对模拟数据生成length、link和序列结果，保存格式为
  
    dict: {'Length': [], 'Link': [], 'Sequence': []}，其中Link和Sequence为二维list，Link对应[-1, 0, 2, 1, 3]顺序下的最合适的边
  
    补充：[-1, 0, 2, 1, 3]分别表示司机、乘客1上车、乘客2上车、乘客1下车、乘客2下车
  
- data
  - bj_link_info_add_geo_2022110912_ds: 北京的路网数据
  - link_feature: 北京二环内的道路特征信息，含80700条字段，共79531条边，62622个节点
  - sample_order.txt: 生成的模拟订单数据

Note:
- bj_link_info中给出的是有向道路，以'0/1'结尾；link_feature给出的是无向道路；因此在生成模拟订单时需要进行匹配
