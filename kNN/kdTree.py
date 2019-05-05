# 对构建好的kd树进行搜索，寻找与目标点最近的样本点：
from math import sqrt
from collections import namedtuple
import numpy as np

# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple", "nearest_point  nearest_dist  nodes_visited  node_type")


# kd-tree每个结点中主要包含的数据结构如下
class KdNode(object):
    def __init__(self, dom_elt, split, left, right, flag):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号）
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree
        self.flag = flag


class KdTree(object):
    def __init__(self, data, label):
        k = len(data[0])  # 数据维度

        def CreateNode(split, data_set):  # 按第split维划分数据集exset创建KdNode
            if not data_set:  # 数据集为空
                return None
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
            # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
            # data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2  # //为Python中的整数除法
            median = data_set[split_pos]  # 中位数分割点
            split_next = (split + 1) % k  # cycle coordinates

            # 递归的创建kd树
            return KdNode(median, split,
                          CreateNode(split_next, data_set[:split_pos]),  # 创建左子树
                          CreateNode(split_next, data_set[split_pos + 1:]), label[split_pos])  # 创建右子树

        self.root = CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)


def find_nearest(tree, point, count=1):
    k = len(point)  # 数据维度
    knears = {}
    global pointlist  # 存储排序后的k近邻点和对应距离

    def travel(kd_node, target):
        if kd_node is None:
            return result([0] * k, float("inf"), 0, -1)  # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1
        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))
        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            travel(kd_node.left, target)
        else:  # 目标离右子树更近
            travel(kd_node.right, target)
        if len(knears) < k:
            knears.setdefault(tuple(kd_node), temp_dist)
            pointlist = sorted(knears.items(), key=lambda item: item[1], reverse=True)
        elif temp_dist <= pointlist[0][1]:
            knears.setdefault(tuple(kd_node), temp_dist)
            pointlist = sorted(knears.items(), key=lambda item: item[1], reverse=True)
        if kd_node.right != None or kd_node.left != None :

    return pointlist

    return travel(tree.root, point)  # 从根节点开始递归


# 搜索树：输出目标点的近邻点
def traveltree(node, aim):
    global pointlist  # 存储排序后的k近邻点和对应距离
    if node == None: return
    col = node.col
    if aim[col] > node.value[col]:
        traveltree(node.rb, aim)
    if aim[col] < node.value[col]:
        traveltree(node.lb, aim)
    dis = dist(node.value, aim)
    if len(knears) < k:
        knears.setdefault(tuple(node.value.tolist()), dis)  # 列表不能作为字典的键
        pointlist = sorted(knears.items(), key=lambda item: item[1], reverse=True)
    elif dis <= pointlist[0][1]:
        knears.setdefault(tuple(node.value.tolist()), dis)
        pointlist = sorted(knears.items(), key=lambda item: item[1], reverse=True)
    if node.rb != None or node.lb != None:
        if abs(aim[node.col] - node.value[node.col]) < pointlist[0][1]:
            if aim[node.col] < node.value[node.col]:
                traveltree(node.rb, aim)
            if aim[node.col] > node.value[node.col]:
                traveltree(node.lb, aim)
    return pointlist


def dist(x1, x2):  # 欧式距离的计算
    return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5