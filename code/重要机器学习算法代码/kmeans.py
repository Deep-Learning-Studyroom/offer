import numpy as np
import random

def kmeans(rows, k):
    """
    伪代码：
    function K-Means(输入数据，中心点个数K)
    获取输入数据的维度Dim和个数N
    随机生成K个Dim维的点
    while(算法未收敛)
        对N个点：计算每个点属于哪一类。
        对于K个中心点：
            1，找出所有属于自己这一类的所有数据点
            2，把自己的坐标修改为这些数据点的中心点坐标
    end
    输出结果：
    end

    算法复杂度：
        1. 时间复杂度
        2. 空间复杂度

    """


    # 确定每个点的最大值和最小值，给随机数定个范围
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows]))
              for i in range(len(rows[0]))]

    # 随机建立k个中心点
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                 for i in range(len(rows[0]))] for j in range(k)]

    lastmatches = None
    # 设定循环100次，也可以定义为函数的一个输入
    for t in range(100):
        print('Iteration {}'.format(t))
        bestmatches = [[] for i in range(k)]

        # 在每一行中寻找距离最近的中心点
        for j, row in enumerate(rows):
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row): bestmatch = i
            bestmatches[bestmatch].append(j)

            # 如果结果与上一次的相同，则整个过程结束
        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches

        # 将中心点移到其所有成员的平均位置处
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs

    return bestmatches

def distance(array1, array2):
    # 判断两个数据的维度是否一致
    if array1.shape != array2.shape:
        raise ValueError("输入数据维度不一致")

    return np.sqrt((array1 - array2) ** 2)