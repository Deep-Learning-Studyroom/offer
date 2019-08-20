import numpy as np
import random

def kmeans(data, k):
    m = len(data)     # 样本个数
    n = len(data[0])  # 维度
    cluster_center = np.zeros((k, n))   # k个聚类中心

    # 选择合适的初始聚类中心
    # 在已有数据中随机选择聚类中心
    # 也可以直接用随机的聚类中心


    init_list = np.random.randint(low=0, high=m, size=k)    # [0, m)
    for index, j in enumerate(init_list):
        cluster_center[index] = data[j][:]

    # 聚类
    cluster = np.zeros(m, dtype=np.int) - 1 # 所有样本尚未聚类
    cc = np.zeros((k, n))   # 下一轮的聚类中心
    c_number = np.zeros(k)    # 每个簇样本的数目

    for times in range(40):
        for i in range(m):
            c = nearst(data[i], cluster_center)
            cluster[i] = c  # 第i个样本归于第c簇
            c_number[c] += 1
            cc[c] += data[i]
        for i in range(k):
            cluster_center[i] = cc[i] / c_number[i]
        cc.flat = 0
        c_number.flat = 0
    return cluster

def nearst(data, cluster_center):
    nearst_center_index = 0
    dis = np.sum((cluster_center[0] - data) ** 2)
    for index, center in enumerate(cluster_center):
        dis_temp = np.sum((center - data) ** 2)
        if dis_temp < dis:
            nearst_center_index = index
            dis = dis_temp
    return nearst_center_index





