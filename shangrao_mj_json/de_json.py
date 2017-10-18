# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : de_json.py
@time : 2017/9/24 9:54

"""
import json
import re
import os
from pprint import pprint

with open('26477270.json', encoding='utf-8') as json_file:
    data = json.load(json_file)
'''
for line in open('26477270.json', encoding='utf-8'):
    dicList.append(json.loads(line))
'''
# print(data)
#print(data.keys)
hu_seat_id_str = str(data['hu_seat_id'])
T = []    # T is action list array
L = []    # temporary array
I = []    # init hands tiles

for key in data:
    # print(data[key])
    if isinstance(data[key], list):
    # pending the type is list
      # print(data[key])
      # print(type(data[key]))
        L.append(data[key])
    # if isinstance(data[key], int):
       # print(data[key])
       # print(type(data[key]))
        # I.append(data[key])

# debug print(data['init_cards'][hu_seat_id_str])
# = I[1]    # debug print(I[1]) I[1] represents hu_seat_id
# hu_seat_string = str(hu_seat)   # transform to string
T = L[3].copy()        # define the players actions
cal_T = len(T)-1
I = data['init_cards'][hu_seat_id_str]
# debug print(T[104])

for cal in reversed(range(cal_T)):
    # L[3] represents actions of players
    # debug print(T[cal])
    # debug print(T[cal][0])
    # debug print(type(T[cal]))
    t = int(T[cal][0])
    if t != data['hu_seat_id']:
        # save the winner actions
        del T[cal]

I_split = []        # I_split is the list of  init hands tiles
I_split = re.findall('[1-9]+[mps]|[ijkwxyz]', I)
print(I_split)  # debug
cal_T = len(T)-1

for cal in range(cal_T):
    print(T[cal][1:4])
    if T[cal][1] == 'G':
        print('G bingo')
        # I_split.append(T[cal][2:4])
    if T[cal][1] == 'd':
        if T[cal][2:4] in I_split:
            print(I_split.index(T[cal][2:4] ))
            # I_split.pop(T[cal][2:4])

print(T)
I_split.sort()
print(I_split)

# def catch_cards(str):
"""

path = "D:\shangraomjpb"     # 文件夹目录
files = os.listdir(path)    # 得到文件夹下的所有文件名称
s = []
for file in files:          # 遍历文件夹
     if not os.path.isdir(file):        # 判断是否是文件夹，不是文件夹才打开
          f = open(path+"/"+file, encoding='utf-8')       # 打开文件
          iter_f = iter(f)      # 创建迭代器
          str = ""
          for line in iter_f:       # 遍历文件，一行行遍历，读取文本
              str = str + line
          s.append(str)         # 每个文件的文本存到list中
print(s)        # 打印结果
"""




