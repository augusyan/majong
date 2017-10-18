# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : load_json.py
@time : 2017/9/25 15:48

"""
# -*- coding: utf-8 -*-
import json
import time
import re
import os
from pprint import pprint


def extract_winner_hands(data):

    print(data)
    print(data.keys)
    T = []  # list array
    L = []  # temporary array
    I = []  # int array

    for key in data:
        # print (data[key])
        if isinstance(data[key], list):
            # pending the type is list
            # print(data[key])
            # print(type(data[key]))
            L.append(data[key])
        if isinstance(data[key], int):
            # print(data[key])
            # print(type(data[key]))
            I.append(data[key])

    hu_seat = I[1]  # debug print(I[1]) I[1] represents hu_seat_id
    hu_seat_string = str(hu_seat)  # transform to string
    T = L[3].copy()  # define the players actions
    cal_a = len(T) - 1
    # debug print(T[104])

    for cal in reversed(range(cal_a)):
        # L[3] represents actions of players
        # debug print(T[cal])
        # debug print(T[cal][0])
        # debug print(type(T[cal]))
        # debug print(type(temp))
        t = int(T[cal][0])
        if t != hu_seat:
            # save the winner actions
            del T[cal]
    print(T)
    print(data['hu_seat_id'])


def store(data):

    with open('data.json', 'w') as json_file:
        json_file.write(json.dumps(data))


def load():
    with open('data.json', encoding='utf-8') as json_file:
        data = json.load(json_file)
        return data


if __name__ == "__main__":

    data = {}
    data["last"]=time.strftime("%Y%m%d")
    store(data)

    data = load()
    print (data["last"])
    path = "D:\shangraomjpb"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s = []
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = open(path + "/" + file, encoding='utf-8')  # 打开文件
            iter_f = iter(f)  # 创建迭代器
            str = ""
            for line in iter_f:  # 遍历文件，一行行遍历，读取文本
                str = str + line
            s.append(str)  # 每个文件的文本存到list中
    print(s)  # 打印结果