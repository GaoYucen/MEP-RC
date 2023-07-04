# MEP-RC

Structure:
- code
  - data_con.py: 根据北京路网数据以uniform random的方式选择多侯选边，形成模拟订单

- data
  - bj_link_info_add_geo_2022110912_ds: 北京的路网数据
  - link_feature: 北京二环内的道路特征信息
  - sample_order.txt: 生成的模拟订单数据

Note:
- bj_link_info中给出的是有向道路，以'0/1'结尾；link_feature给出的是无向道路；因此在生成模拟订单时需要进行匹配
