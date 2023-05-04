#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import numpy as np
#获取数据-----------------------------------------
#读取安德森鸢尾花卉数据集
iris = datasets.load_iris()
#第50-150个数据(线性不可分)的x向量中的第3,4特征
x2 = iris.data[50:150, [2, 3]]
y2 = iris.target[50: 150]
#更改y2标签为-1或1
y2 = np.where(y2 == 1, -1, 1)

def d(x):
    return np.dot(w,x) + b

#画图部分-----------------------------------------
fig = plt.figure()
fig.canvas.set_window_title("感知机") #修改fig的名字
# fig.suptitle("安德森鸢尾花卉线性不可分数据集",fontproperties='Microsoft YaHei') #由于后面会clf()所以放这里是无效的
cmap = ListedColormap(('darkseagreen', 'blue', 'red'))

w, b = np.array([0, 0]), 0
exist_wrong = True
while exist_wrong:
    exist_wrong = False
    for x, y in zip(x2, y2):
        if y*d(x) <= 0:
            w = w + y*x
            b = b + y
            exist_wrong = True
            break
    plt.clf()
    fig.suptitle("安德森鸢尾花卉线性不可分数据集", fontproperties='Microsoft YaHei') #放这里才有效
    plt.xlim(2.5, 7.5)  # 限制x轴y轴
    plt.ylim(0.5, 3)
    for i in range(50):
        plt.scatter(x2[i][0], x2[i][1], color=cmap(1), edgecolors='k', marker='x', alpha=0.8)
        # edgecolors='k'设置散点边缘颜色为黑,alpha为透明度,marker为散点形状
    for i in range(50, 100):
        plt.scatter(x2[i][0], x2[i][1], color=cmap(2), edgecolors='k', marker='o', alpha=0.8)
    plt.plot((2.5, 7), ((-b - 2.5 * w[0]) / w[1], (-b - 7 * w[0]) / w[1]))
    plt.pause(0.1)

plt.show()
