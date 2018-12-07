# coding: utf-8

import sys
import numpy as np
from collections import Counter

# 获取dataset，决策树不一定是二叉树，每一个属性的值可以有多个，本例中0，1为特例
def getDataset():
    # 属性和最终分类，确定是否为鱼类，最终类型仅有两种是鱼类和非鱼类
    # no surfacing    flippers      fish
    #      1              1          yes
    #      1              1          yes
    #      1              0           no
    data = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]
    args = ['no surfacing', 'flippers']     #  column index  [ 0, 1 ]
    labels = ['fish', 'not fish']           # label index [1,  0]

    dataset = np.array(data)

    return dataset, args, labels

def calcShannonEntropy(dataset):
    data_rows = dataset.shape[0]
    cnt = Counter([row[-1] for row in dataset])
    print('cnt={}'.format(cnt))
    shannonentropy = 0.0
    for label, num in cnt.items():
        pi = num/data_rows
        shannonentropy -=  pi * np.log(pi, 2)

    return shannonentropy

# 根据给定的dataset，按照指定列的列索引进行切割，返回该列的值等于value的子集
def splitDataset(dataset, index, value):
    sublist = [row.tolist() for row in dataset if row[index] == value]
    return np.array(sublist)


# 根据给定的dataset， 如果按照指定的index列对数据进行切割，则返回切割后的
# 子数据集的熵, 子数据集的熵为各个子集的熵的加权平均值
def calcSubShannonEntropy(dataset, index):
    total_rows = dataset.shape[0]
    # 得到某列的去重后的所有值的list
    valset = set(dataset[:][index].tolist())
    weightShannoEntropy = 0.0
    for v in valset:
        subdataset = splitDataset(dataset, index, v)
        # 加权平均
        weight_percentage = subdataset.shape[0]/total_rows
        weightShannoEntropy += weight_percentage * calcShannonEntropy(subdataset)

    return weightShannoEntropy

# 对给定的数据集，选出信息增益最大的一列用于切割数据集
# 输入为数据集，返回为列索引
def chooseBestColumnToSplit(dataset):
    # 计算原始数据集的熵
    origin_entropy = np.array([calcShannonEntropy(dataset)])
    # 最后一列为分类，不是属性
    index_number = dataset.shape[1] - 1
    # 遍历每一列并计算以该列做切割的子集的熵，存成array
    subShannonEntropy_set = np.array([calcSubShannonEntropy(dataset, idx) for idx in range(0, index_number)])
    origin_entropy_set = np.tile(origin_entropy, (1, index_number))
    # 计算与原数据集的熵之间的差值
    ret_set = abs(origin_entropy_set - subShannonEntropy_set)

    # 降序排列，返回第一个索引即为信息增益最大的索引
    return np.argsort(-ret_set)[0]

# 创建决策树

def main(argv=None):
    if not argv:
        argv = sys.argv

    dataset, args_col, labels = getDataset()
    print('dataset.shape={}'.format(dataset.shape))

    calcShannonEntropy(dataset)

if __name__ == '__main__':
    sys.exit(main())