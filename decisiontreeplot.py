# coding: utf-8

# 本模块实现的绘制决策树的方法采用后序遍历
# 在绘制过程中动态计算各个节点之间的间距防止节点覆盖
# 本例的时间复杂的为O（N）相比git上示例代码的O（3N）优化
# 本例在绘制之前不需要计算树的宽度和广度

import matplotlib.pyplot as plt

# 决策节点的样式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
# 分类节点的样式
leafNode = dict(boxstyle="round4", fc="0.8")
# 树节点之间箭头的样式
arrow_args = dict(arrowstyle="->")

# 决策树的根节点的位置
root = (10, 10)
node_x_padding = 3
tree_padding = node_x_padding * 0.1
node_y_padding = 1

def plotLeafNode(text, x_position, y_position):
    print('text {}, x {}, y {}'.format(text, x_position, y_position))
    # createPlot.splot.scatter(x_position, y_position)
    createPlot.splot.annotate(text, xy=(x_position, y_position), xytext=(x_position, y_position),
                              bbox=leafNode)
    return (x_position, y_position)

def plotRoot(text, x_position, y_position, nodes, vals):
    print('text {}, x {}, y {}'.format(text, x_position, y_position))
    # createPlot.splot.scatter(x_position, y_position)
    for idx, node in enumerate(nodes):
        createPlot.splot.annotate(text, xy=node, xytext=(x_position, y_position),
                                  bbox=decisionNode, arrowprops=arrow_args)
        xMid = (node[0] + x_position)/2
        yMid = (node[1] + y_position)/2
        createPlot.splot.text(xMid, yMid, vals[idx], va="center", ha="center", rotation=30)
    print('nodes {}'.format(nodes))

    return (x_position, y_position)

# 后序遍历绘制树,左节点，右节点，树根节点
# 绘制时提供该树的最左边界，绘制结束时返回该树的最右边界
def plotTree(dt, left_edge, y_position):
    root_x = left_edge
    root_y = y_position
    nodes = []
    vals = []
    # 遍历子节点
    if 'leafs' in dt:
        # 记录子节点的x坐标范围，其父节点的位置为该范围的中间
        left_start = None
        left_end = None
        for leaf in dt['leafs']:
            for k, v in leaf.items():
                if isinstance(v, dict):
                    left_edge, node_position = plotTree(v, left_edge, y_position - node_y_padding)
                else:
                    node_position = plotLeafNode(v, left_edge, y_position - node_y_padding)
                    left_edge = left_edge + node_x_padding
                nodes.append(node_position)
                vals.append(k)
                # 存第一个子节点的x位置
                if left_start == None:
                    left_start = node_position[0]
                left_end = node_position[0]
        if left_start is not None:
            root_x = round((left_start + left_end)/2, 2)

    if dt['arg'] == 'any':
        node_position = plotLeafNode(dt['label'], root_x, root_y)
        left_edge += node_x_padding
    else:
        node_position = plotRoot(dt['arg'], root_x, root_y, nodes, vals)

    # 0.3 为同一层树节点的子节点组成的集合之间的边距
    # 例如 A B 为相邻的兄弟节点他们的子节点分别为C,D和E,F
    # 则0.3为D和E之间的边距
    return left_edge + tree_padding, node_position

def initplot():
    fig = plt.figure()
    fig.clf()
    createPlot.splot = None
    createPlot.splot = plt.subplot(111)
    plt.axis([0, 12, 0, 12])

# 本例绘制树显示的分类结果为0，1，未经转换，此处若需要转换，则应该在生成树时直接存储分类结果
# 为对应的fish  or   not fish
def createPlot(dt, left_edge, y_position):
    initplot()
    # createPlot.ax1.annotate('test', xy=(2,2), xytext=(0.9, 0.9), bbox=decisionNode, arrowprops=arrow_args)
    plotTree(dt, left_edge, y_position)
    plt.show()