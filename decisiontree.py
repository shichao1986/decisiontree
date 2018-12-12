# coding: utf-8

import sys
import copy
import numpy as np
import json
import pickle
from collections import Counter
import decisiontreeplot as dtplot

class MyExcept(Exception):
    def __init__(self, msg):
        self.msg = msg

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
    labels = ['not fish', 'fish']           # label index [1,  0]

    dataset = np.array(data)

    return dataset, args, labels

def calcShannonEntropy(dataset):
    data_rows = dataset.shape[0]
    cnt = Counter([row[-1] for row in dataset])
    print('cnt={}'.format(cnt))
    shannonentropy = 0.0
    # import pdb;pdb.set_trace()
    for label, num in cnt.items():
        pi = num/data_rows
        # numpy 内部没有任意数为底数的对数公式，如果需要使用任意底数的对数公式可以有两种方式
        # 1， import math;   math.log(3,4) 意为以3为底4的对数
        # 2，根据对数换底公式  math.log(3,4) = math.log(X,4) / math.log(X,3)，所以
        # 以3为底4的对数等于   np.log(4) / np.log(3)
        shannonentropy -=  pi * np.log2(pi)

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
    valset = set(dataset[:,index].tolist())
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
    origin_entropy = calcShannonEntropy(dataset)    #np.array([calcShannonEntropy(dataset)])
    # 最后一列为分类，不是属性
    index_number = dataset.shape[1] - 1
    # 遍历每一列并计算以该列做切割的子集的熵，存成array
    subShannonEntropy_set = np.array([calcSubShannonEntropy(dataset, idx) for idx in range(0, index_number)])
    origin_entropy_set = np.ones(2) *  origin_entropy   #np.tile(origin_entropy, (1, index_number))
    # 计算与原数据集的熵之间的差值
    ret_set = abs(origin_entropy_set - subShannonEntropy_set)

    # 降序排列，返回第一个索引即为信息增益最大的索引
    # import pdb;pdb.set_trace()
    return np.argsort(-ret_set)[0]

# 创建决策树
# 树的数据结构可以用类实现也可以用字典实现，本例用字典实现
# 字段内容：
''' TREE_DICT:
    {
        arg: 属性名称(值为any时直接取label为结果，否则继续遍历leafs)
        leafs:[{value: 属性可取值1, node:TREE_DICT},
                {value: 属性可取值2, node:TREE_DICT]
        label:分类结果
    }
'''
# 入参为dataset，arglabel 返回值为 TREE_DICT
def createTree(dataset, col_labels):
    arglabel = copy.deepcopy(col_labels)
    tree_dict = {}
    # 先判断最终分类是否仅有一种，若如此则直接返回分类
    label_list = dataset[:,-1].tolist()
    if len(set(label_list)) == 1:
        tree_dict['arg'] = 'any'
        tree_dict['leafs'] = []
        tree_dict['label'] = label_list[0]
        return tree_dict

    # 再判断属性列是否为0，若为0，则返回标签列中第一个
    # 出现的最多次数的分类（最好的情况是当属性列为0时，所有的分类均相同）
    # 一些其他算法在此处判断属性列是否为1，如果为1则不再继续分割，这是因为
    # 该类算法实现的决策树的分类结果存储在叶子节点上，当列数量为1时，叶子节点
    # 直接根据不通的属性值存取占比最多的结果；而此列中的分类结果存储在页节点，
    # 所以结束的条件为属性列为0
    if dataset.shape[1] == 1: #仅剩的最后一列标签列
        tree_dict['arg'] = arglabel[0]  #仅剩最后一个label
        tree_dict['leafs'] = []
        tree_dict['label'] = Counter(dataset[:,-1].tolist()).most_common(1)[0][0]
        return tree_dict
    # 上述两种情况不满足，则需要继续创建子树
    #  根据函数chooseBestColumnToSplit 获取本节点的属性值（arg: 属性名称）
    #  遍历该列的不同值生成leafs
    '''
        d = {}
        d['value'] = 属性可取值1
        d['node'] = createTree(subdataset, subarglabel)
        TREE_DICT['leafs'].append(d)

    '''
    choose_idx = chooseBestColumnToSplit(dataset)
    tree_dict['arg'] = arglabel[choose_idx]
    arglabel.pop(choose_idx)
    tree_dict['label'] = None
    tree_dict['leafs'] = []
    choose_vals = set(dataset[:,choose_idx].tolist())
    # import pdb;pdb.set_trace()
    for val in choose_vals:
        d = {}
        d['value'] = val
        leaf_dataset = splitDataset(dataset, choose_idx, val)
        # 删除第choose_idx列
        leaf_dataset = np.delete(leaf_dataset, choose_idx, 1)
        d['node'] = createTree(leaf_dataset, arglabel)
        tree_dict['leafs'].append(d)

    return tree_dict

# 遍历决策树，对测试数据进行分类
def DTwalk(test_dataset, arglabel, dt):
    if dt['label'] is not None:
        return dt['label']

    # 求出决策树当前属性名称在arglabel中的列 col
    # 求出test_dataset中col列的属性值为多少
    walk_val = test_dataset[arglabel.index(dt['arg'])]
    for sub in dt['leafs']:
        if sub['value'] == walk_val:
            return DTwalk(test_dataset, arglabel, sub['node'])

    raise MyExcept('error decision tree, lost case of val = {}, of arg:{}'.format(walk_val, dt['arg']))

def DTwalk2(test_dataset, arglabel, dt):
    if dt['arg'] == 'any':
        return dt['label']
    walk_val = test_dataset[arglabel.index(dt['arg'])]
    for node in dt['leafs']:
        if walk_val in node:
            if isinstance(node[walk_val], dict):
                return DTwalk2(test_dataset, arglabel, node[walk_val])
            else:
                return node[walk_val]

    raise MyExcept('error decision tree, lost case of val = {}, of arg:{}'.format(walk_val, dt['arg']))

# 创建决策树的第二种方法，仅树的数据结构略微不同
def createTree2(dataset, arglabel):
    tree_dict = {}
    label_list = dataset[:,-1].tolist()
    if len(set(label_list)) == 1:
        tree_dict['arg'] = 'any'
        tree_dict['label'] = label_list[0]
        return tree_dict
    col_num = dataset.shape[1]
    if col_num == 1+1:
        tree_dict['arg'] = arglabel[0]
        tree_dict['leafs'] = []
        vals = set(dataset[:,0].tolist())
        for val in vals:
            d = {}
            subdataset = splitDataset(dataset, 0, val)
            d[val] = Counter(subdataset[:,-1].tolist()).most_common(1)[0][0]
            tree_dict['leafs'].append(d)
        return tree_dict
    choose_idx = chooseBestColumnToSplit(dataset)
    tree_dict['arg'] = arglabel[choose_idx]
    arglabel.pop(choose_idx)
    tree_dict['leafs'] = []
    vals = set(dataset[:, choose_idx].tolist())
    for val in vals:
        d = {}
        subdataset = splitDataset(dataset, 0, val)
        subdataset = np.delete(subdataset, choose_idx, 1)
        d[val] = createTree2(subdataset, arglabel)
        tree_dict['leafs'].append(d)
    return tree_dict

def main(argv=None):
    if not argv:
        argv = sys.argv

    dataset, args_col, labels = getDataset()
    test_col = copy.deepcopy(args_col)
    print('dataset.shape={}'.format(dataset.shape))

    dt = createTree(dataset, args_col)
    dt2 = createTree2(dataset, args_col)

    try:
        test_dataset = np.array([1,0])
        # import pdb; pdb.set_trace()
        classfy = DTwalk(test_dataset, test_col, dt)
        classfy2 = DTwalk2(test_dataset, test_col, dt2)
        print('dt is:{}'.format(dt))
        print('dt2 is:{}'.format(dt2))
        print('classfy result is:{}'.format(labels[classfy]))
        print('classfy2 result is:{}'.format(labels[classfy2]))
        # import pdb;pdb.set_trace()
        dtplot.createPlot(dt2, 0, 10)

        # 持久化
        # json 与 pickle对比，
        # json 更为通用，且仅支持python基本数据类型，不支持datetime， function 等，多用于http业务
        # pickle 提供python所有数据类型的序列化
        # print('json data:{}'.format(json.dumps(dt)))
        # print('json obj:{}'.format(json.loads(json.dumps(dt))))
        # print('pickle data:{}'.format(pickle.dumps(dt)))
        # print('pickle obj:{}'.format(pickle.loads(pickle.dumps(dt))))
    except MyExcept as e:
        print(e)
        return 2

    return 0

if __name__ == '__main__':
    sys.exit(main())