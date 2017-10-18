# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : train_model_Bayes_demo.py
@time : 2017/10/4 8:36
@function : 单人麻将的训练模型
"""
import json
import re
import os
import tensorflow as tf
import keras

class NaiveBayesMethod:
    '''
    NaiveBayesMethod 的内部计算方式现在为数值计算,
    符号计算的代码已经注释,如果需要请手动修改

    朴素贝叶斯法分类器 当lam=1 时,类分类方式为为贝叶斯估计
    实现了拉普拉斯平滑,以此避免出现要计算的概率为0的情况,以免计算错误的累积
    具体原理请参考李航的<统计学习方法>第四章
    lam = 0 时 类分类方式为为极大似然值估计
    '''

    def __init__(self, inputArray, lam):
        self.input = inputArray
        self.lam = lam
        self.__lenInput = len(self.input)
        self.__y = self.input[self.__lenInput - 1]
        self.__onlyy = self.__only(self.__y)
        self.__county = self.__countList(self.__onlyy)

    # 计算列表总样本数 return int
    def __countList(self, list):
        count = {}
        for item in list:
            count[item] = count.get(item, 0) + 1
        return len(count)

    # 检查某列表中时候含有某个元素
    def __findy(self, list, y):
        result = True
        for i in range(0, len(list)):
            if list[i] == y:
                result = False
        return result

    # 返回列表种类
    def __only(self, list):
        onlyy = []
        for i in range(0, len(list)):
            if self.__findy(onlyy, list[i]):
                onlyy.append(list[i])
        return onlyy

    # 统计列表中某元素的个数
    def __countKind(self, list, element):
        return list.count(element)

    #  通过元素值返回位置索引
    def __findOnlyElement(self, list, x):
        return self.__only(list).index(x)

    # 先验概率
    def __py(self, x):
        # return Fraction(self.__countKind(self.__y, x) + self.lam, len(self.__y) + self.__county * self.lam)
        return (self.__countKind(self.__y, x) + self.lam) / (len(self.__y) + self.__county * self.lam)

    # 返回p(x=?)
    def __probabilityX(self, list, x):
        # return Fraction(self.__countKind(list, x) + self.lam, len(list) + self.__countList(list) * self.lam)
        return (self.__countKind(list, x) + self.lam) / (len(list) + self.__countList(list) * self.lam)

    def __probabilityYX(self, list, x, yy):
        xx = self.__findOnlyElement(list, x)
        yindex = self.__findOnlyElement(self.__y, yy)
        fz = 0
        onlyx = self.__only(list)
        onlyy = self.__only(self.__y)
        # 获取 p(y=?|x1=?) 的分子
        for i in range(0, len(list)):
            if list[i] == onlyx[xx] and self.__y[i] == onlyy[yindex]:
                fz += 1
        # return Fraction(fz + self.lam, self.__countKind(list, onlyx[xx]) + self.__countList(list) * self.lam)
        return (fz + self.lam) / (self.__countKind(list, onlyx[xx]) + self.__countList(list) * self.lam)

    def fl(self, x, y):
        ps = []
        for i in range(0, len(self.__onlyy)):
            p1 = self.__probabilityX(self.input[0], x) * self.__probabilityYX(self.input[0], x,
                                                                              1) * self.__probabilityX(
                self.input[1], y) * self.__probabilityYX(self.input[1], y, self.__onlyy[i]) / self.__py(1)
            ps.append(p1)
        return self.__onlyy[ps.index(max(ps))]


# 测试NaiveBayesMethod
input = [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
         [1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3, 3],
         [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
test = NaiveBayesMethod(input, 1)
print(test.fl(2, 1))
test.lam = 0
print(test.fl(2, 1))

