# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : feature_extract_v2.py
@time : 2017/10/15 15:14
@function : 
"""
import json
import re
import os


def wait_types_comm(tile_list):
    common_waiting = {'common_waiting0': 0,
                      'common_waiting1': 0,
                      'common_waiting2': 0,
                      'common_waiting3': 0,
                      'common_waiting4': 0,
                      'common_waiting5': 0,
                      'common_waiting6': 0,
                      'common_waiting7': 0,
                      'common_waiting8': 0,
                      'common_waiting9': 0,
                      'common_waiting10': 0,
                      'common_waiting11': 0,
                      'common_waiting12': 0,
                      'common_waiting13': 0,
                      }
    tempList = tile_list
    # print tempList
    wait_num = 14

    sz = 0  # 顺子数
    kz = 0  # 刻子数
    dzk = 0  # 搭子 aa
    dzs12 = 0  # 搭子ab
    dzs13 = 0  # 搭子ac
    # 判断顺子数
    i = 0
    while i <= len(tempList) - 3:
        if tempList[i] & 0xF0 != 0x30 and tempList[i] + 1 in tempList and tempList[i] + 2 in tempList:
            # print(tempList[i], "i")
            wait_num -= 3
            sz += 1
            card0 = tempList[i]
            card1 = tempList[i] + 1
            card2 = tempList[i] + 2
            tempList.remove(card0)
            tempList.remove(card1)
            tempList.remove(card2)
        else:
            i += 1
    # print(tempList)

    # 判断刻子数
    j = 0
    while j <= len(tempList) - 3:
        if tempList[j + 1] == tempList[j] and tempList[j + 2] == tempList[j]:
            # print(tempList[j], "j")
            wait_num -= 3
            kz += 1
            card = tempList[j]
            tempList.remove(card)
            tempList.remove(card)
            tempList.remove(card)
        else:
            j += 1
    # print(tempList)

    # 判断搭子aa
    x = 0
    while x <= len(tempList) - 2:
        # print(tempList[x], "x")
        if tempList[x + 1] == tempList[x]:
            dzk += 1
            # wait_num -=2
            card = tempList[x]
            tempList.remove(card)
            tempList.remove(card)
        else:
            x += 1

    # 判断搭子ab ac
    k = 0
    while k <= len(tempList) - 2:
        if tempList[k] & 0xF0 != 0x30:
            # print(tempList[k], "k")
            if tempList[k] + 1 in tempList:
                # wait_num -= 2
                dzs12 += 1
                card0 = tempList[k]
                card1 = tempList[k] + 1
                tempList.remove(card0)
                tempList.remove(card1)
            elif tempList[k] + 2 in tempList:
                # wait_num -= 2
                dzs13 += 1
                card0 = tempList[k]
                card2 = tempList[k] + 2
                tempList.remove(card0)
                tempList.remove(card2)
            else:
                k += 1
        else:
            k += 1
    if dzk > 0:  # 如果搭子aa>0 ,取其中一个作为将牌，并且向听数-2
        wait_num -= 2
        if dzk - 1 + dzs12 + dzs13 - (4 - sz - kz) <= 0:  # 如果搭子加面子<=4,向听数再减去搭子数*2
            wait_num -= (dzk - 1 + dzs12 + dzs13) * 2
        else:
            wait_num -= (4 - sz - kz) * 2  # 否则 向听数只减去多余的，即向听数减到为0
    else:  # 如果搭子aa=0，取一张单牌作为将的候选，向听数-1
        wait_num -= 1
        if dzk + dzs12 + dzs13 - (4 - sz - kz) <= 0:  # 向上同理
            wait_num -= (dzk + dzs12 + dzs13) * 2
        else:
            wait_num -= (4 - sz - kz) * 2
    # print(tempList)

    common_waiting['common_waiting' + str(wait_num)] = 1
    return wait_num
    # print(common_waiting)


def wait_types_7(tile_list):
    # 七对的向听数判断
    wait_7couples = {
        'seven_waiting0': 0,
        'seven_waiting1': 0,
        'seven_waiting2': 0,
        'seven_waiting3': 0,
        'seven_waiting4': 0,
        'seven_waiting5': 0,
        'seven_waiting6': 0,
        'seven_waiting7': 0,
    }
    wait_num = 7    # 表示向听数
    tile_list.sort()    # L是临时变量，传递tile_list的值
    L = set(tile_list)
    for i in L:
        # print("the %d has %d in list" % (i, tile_list.count(i)))
        if tile_list.count(i) >= 2:
            wait_num -= 1
    # print(tile_list)
    # wait_types_7['seven_waiting'+str(wait_num)] = 1
    # print(wait_num)
    wait_7couples['seven_waiting'+str(wait_num)] = 1
    # print(wait_7couples)
    return wait_num


def wait_types_13(tile_list):
    # 十三浪的向听数判断，手中十四张牌中，序数牌间隔大于等于3，字牌没有重复所组成的牌形
    # 先计算0x0,0x1,0x2中的牌，起始位a，则a+3最多有几个，在wait上减，0x3计算不重复最多的数
    wait_13lan = {
        'thirteen_waiting0': 0,
        'thirteen_waiting1': 0,
        'thirteen_waiting2': 0,
        'thirteen_waiting3': 0,
        'thirteen_waiting4': 0,
        'thirteen_waiting5': 0,
        'thirteen_waiting6': 0,
        'thirteen_waiting7': 0,
        'thirteen_waiting8': 0,
        'thirteen_waiting9': 0,
        'thirteen_waiting10': 0,
        'thirteen_waiting11': 0,
        'thirteen_waiting12': 0,
        'thirteen_waiting13': 0,
        'thirteen_waiting14': 0,
    }
    wait_num = 14  # 表示向听数
    max_num_wait = 0
    # print(wait_13lan)
    L = set(tile_list)  # 去除重复手牌
    L_num0 = []     # 万数牌
    L_num1 = []     # 条数牌
    L_num2 = []     # 筒数牌
    for i in L:
        if i & 0xf0 == 0x30:
            # 计算字牌的向听数
            wait_num -= 1
        if i & 0xf0 == 0x00:
            L_num0.append(i)
        if i & 0xf0 == 0x10:
            L_num1.append(i)
        if i & 0xf0 == 0x20:
            L_num2.append(i)
    wait_num -= calculate_13(L_num0)
    # 减去万数牌的向听数
    wait_num -= calculate_13(L_num1)
    # 减去条数牌的向听数
    wait_num -= calculate_13(L_num2)
    # 减去筒数牌的向听数
    # print(L)
    # print(L_num0)
    # print(L_num1)
    # print(L_num2)
    # print(wait_num)
    wait_13lan['thirteen_waiting' + str(wait_num)] = 1
    # print(wait_13lan)
    return wait_num


def wait_types_19(tile_list):
    # 九幺的向听数判断，由一、九这些边牌、东、西、南、北、中、发、白这些风字牌中的任意牌组成的牌形。以上这些牌可以重复
    wait_19 = {
        'one_nine_waiting0': 0,
        'one_nine_waiting1': 0,
        'one_nine_waiting2': 0,
        'one_nine_waiting3': 0,
        'one_nine_waiting4': 0,
        'one_nine_waiting5': 0,
        'one_nine_waiting6': 0,
        'one_nine_waiting7': 0,
        'one_nine_waiting8': 0,
        'one_nine_waiting9': 0,
        'one_nine_waiting10': 0,
        'one_nine_waiting11': 0,
        'one_nine_waiting12': 0,
        'one_nine_waiting13': 0,
        'one_nine_waiting14': 0,
    }
    wait_num = 14    # 表示向听数
    tile_list.sort()    # 排序
    L = set(tile_list)  # L是临时变量，传递tile_list的值
    for i in tile_list:
        if i & 0x0f == 0x01 or i & 0x0f == 0x09 or i & 0xf0 == 0x30:
            wait_num -= 1
    wait_19['one_nine_waiting' + str(wait_num)] = 1
    # print(wait_19)
    return wait_num


def calculate_13(tiles):
    # 计算十三浪的数牌最大向听数
    return max((tiles.count(1)+tiles.count(4)+tiles.count(7)), (tiles.count(1)+tiles.count(4)+tiles.count(8)), \
               (tiles.count(1) + tiles.count(4) + tiles.count(9)), (tiles.count(1)+tiles.count(5)+tiles.count(8)), \
               (tiles.count(1) + tiles.count(5) + tiles.count(9)), (tiles.count(2)+tiles.count(5)+tiles.count(8)), \
               (tiles.count(2) + tiles.count(5) + tiles.count(9)), tiles.count(3)+tiles.count(6)+tiles.count(9))

# 2到8牌的个数
def num_tile_2_8(cards):
    num_tile = {'num_tile0': 0,
                'num_tile1': 0,
                'num_tile2': 0,
                'num_tile3': 0,
                'num_tile4': 0,
                'num_tile5': 0,
                'num_tile6': 0,
                'num_tile7': 0,
                'num_tile8': 0,
                'num_tile9': 0,
                'num_tile10': 0,
                'num_tile11': 0,
                'num_tile12': 0,
                'num_tile13': 0,
                'num_tile14': 0
                }  # 0到14
    n = 0
    for c in cards:
        if c & 0x0F > 1 and c & 0x0F < 9:
            ++n
    num_tile['num_tile' + str(n)] = 1
    return n


# 某种花色牌最多的张数
def most_tile_flowerCards(cards):
    most_tile = {'most_tile0': 0,
                 'most_tile1': 0,
                 'most_tile2': 0,
                 'most_tile3': 0,
                 'most_tile4': 0,
                 'most_tile5': 0,
                 'most_tile6': 0,
                 'most_tile7': 0,
                 'most_tile8': 0,
                 'most_tile9': 0,
                 'most_tile10': 0,
                 'most_tile11': 0,
                 'most_tile12': 0,
                 'most_tile13': 0,
                 'most_tile14': 0
                 }  # 0到14
    count = {}
    n = 1
    for i in range(len(cards) - 1):
        if cards[i] & 0xF0 == cards[i + 1] & 0xF0:
            n += 1
            # print n
            if i == len(cards) - 2:
                count[str(cards[i] & 0xF0)] = n
        else:
            count[str(cards[i] & 0xF0)] = n
            n = 1
    if cards[len(cards) - 1] & 0xF0 != cards[len(cards) - 2] & 0xF0:
        count[str(cards[len(cards) - 1] & 0xF0)] = 1
    temp = 0
    for key in count:
        # print key,"=",count[key]
        if key != 3 and temp < count[key]:
            temp = count[key]
    most_tile['most_tile' + str(temp)] = 1
    return temp


# 各色1到9有无
def num_has1to9(cards):  # 1万到9万，一条到9条，一筒到9筒
    has_1to9 = {'num_19_wan_1': 0,
                'num_19_wan_2': 0,
                'num_19_wan_3': 0,
                'num_19_wan_4': 0,
                'num_19_wan_5': 0,
                'num_19_wan_6': 0,
                'num_19_wan_7': 0,
                'num_19_wan_8': 0,
                'num_19_wan_9': 0,
                'num_19_tiao_1': 0,
                'num_19_tiao_2': 0,
                'num_19_tiao_3': 0,
                'num_19_tiao_4': 0,
                'num_19_tiao_5': 0,
                'num_19_tiao_6': 0,
                'num_19_tiao_7': 0,
                'num_19_tiao_8': 0,
                'num_19_tiao_9': 0,
                'num_19_tong_1': 0,
                'num_19_tong_2': 0,
                'num_19_tong_3': 0,
                'num_19_tong_4': 0,
                'num_19_tong_5': 0,
                'num_19_tong_6': 0,
                'num_19_tong_7': 0,
                'num_19_tong_8': 0,
                'num_19_tong_9': 0,
                }
    # print cards
    wan = 0
    tiao = 0
    tong = 0
    for c in cards:
        if c & 0xF0 == 0x00:
            wan += 2 << c & 0x0F
            # has_1to9['num_19_wan_'+str(c&0x0F)]=1
        elif c & 0xF0 == 0x10:
            tiao += 2 << c & 0x0F
            # has_1to9['num_19_tiao_'+str(c&0x0F)]=1
        elif c & 0xF0 == 0x20:
            tong += 2 << c & 0x0F
            # has_1to9['num_19_tong_'+str(c&0x0F)]=1
    has_1to9 = []
    has_1to9.append(wan, tiao, tong)
    return has_1to9


# 备齐3张以上的相同牌的个数
def num_same_3tile(cards):  # 3张以上相同的牌，0到4
    same_3tile = {'same_3tile_0': 0,
                  'same_3tile_1': 0,
                  'same_3tile_2': 0,
                  'same_3tile_3': 0,
                  'same_3tile_4': 0,
                  }
    count = {}
    n = 1
    for i in range(len(cards) - 1):
        if cards[i] == cards[i + 1]:
            n += 1
            print
            n
            if i == len(cards) - 2:
                count[str(cards[i])] = n
        else:
            count[str(cards[i])] = n
            n = 1
    if cards[len(cards) - 1] != cards[len(cards) - 2]:
        count[str(cards[len(cards) - 1])] = 1
    count_same3 = 0
    for key in count:
        # print key,"=",count[key]
        if count[key] >= 3:
            count_same3 += 1

    same_3tile['same_3tile_' + str(count_same3)] = 1
    return count_same3


# 备齐2张以上相同牌的个数
def num_same_2tile(cards):  # 2张以上相同的牌，0到6
    same_2tile = {'same_2tile_0': 0,
                  'same_2tile_1': 0,
                  'same_2tile_2': 0,
                  'same_2tile_3': 0,
                  'same_2tile_4': 0,
                  'same_2tile_5': 0,
                  'same_2tile_6': 0,
                  }
    count = {}
    n = 1
    for i in range(len(cards) - 1):
        if cards[i] == cards[i + 1]:
            n += 1
            # print n
            if i == len(cards) - 2:
                count[str(cards[i])] = n
        else:
            count[str(cards[i])] = n
            n = 1
    if cards[len(cards) - 1] != cards[len(cards) - 2]:
        count[str(cards[len(cards) - 1])] = 1
    count_same2 = 0
    for key in count:
        # print key,"=",count[key]
        if count[key] >= 2:
            count_same2 += 1

    same_2tile['same_3tile_' + str(count_same2)] = 1
    return count_same2


def common_waiting(cards):
    common_waiting = {'common_waiting0': 0,
                      'common_waiting1': 0,
                      'common_waiting2': 0,
                      'common_waiting3': 0,
                      'common_waiting4': 0,
                      'common_waiting5': 0,
                      'common_waiting6': 0,
                      'common_waiting7': 0,
                      'common_waiting8': 0,
                      'common_waiting9': 0,
                      'common_waiting10': 0,
                      'common_waiting11': 0,
                      'common_waiting12': 0,
                      'common_waiting13': 0,
                      }
    tempList = cards
    # print tempList
    wait_num = 14

    sz = 0  # 顺子数
    kz = 0  # 刻子数
    dzk = 0  # 搭子 aa
    dzs12 = 0  # 搭子ab
    dzs13 = 0  # 搭子ac
    # 判断顺子数
    i = 0
    while i <= len(tempList) - 3:
        if tempList[i] & 0xF0 != 0x30 and tempList[i] + 1 in tempList and tempList[i] + 2 in tempList:
            # print tempList[i],"i"
            wait_num -= 3
            sz += 1
            card0 = tempList[i]
            card1 = tempList[i] + 1
            card2 = tempList[i] + 2
            tempList.remove(card0)
            tempList.remove(card1)
            tempList.remove(card2)
        else:
            i += 1
    # print tempList

    # 判断刻子数
    j = 0
    while j <= len(tempList) - 3:
        if tempList[j + 1] == tempList[j] and tempList[j + 2] == tempList[j]:
            # print tempList[j],"j"
            wait_num -= 3
            kz += 1
            card = tempList[j]
            tempList.remove(card)
            tempList.remove(card)
            tempList.remove(card)
        else:
            j += 1
    # print tempList

    # 判断搭子aa
    x = 0
    while x <= len(tempList) - 2:
        # print tempList[x],"x"
        if tempList[x + 1] == tempList[x]:
            dzk += 1
            # wait_num -=2
            card = tempList[x]
            tempList.remove(card)
            tempList.remove(card)
        else:
            x += 1

    # 判断搭子ab ac
    k = 0
    while k <= len(tempList) - 2:
        if tempList[k] & 0xF0 != 0x30:
            # print tempList[k],"k"
            if tempList[k] + 1 in tempList:
                # wait_num -= 2
                dzs12 += 1
                card0 = tempList[k]
                card1 = tempList[k] + 1
                tempList.remove(card0)
                tempList.remove(card1)
            elif tempList[k] + 2 in tempList:
                # wait_num -= 2
                dzs13 += 1
                card0 = tempList[k]
                card2 = tempList[k] + 2
                tempList.remove(card0)
                tempList.remove(card2)
            else:
                k += 1
        else:
            k += 1
    if dzk > 0:  # 如果搭子aa>0 ,取其中一个作为将牌，并且向听数-2
        wait_num -= 2
        if dzk - 1 + dzs12 + dzs13 - (4 - sz - kz) <= 0:  # 如果搭子加面子<=4,向听数再减去搭子数*2
            wait_num -= (dzk - 1 + dzs12 + dzs13) * 2
        else:
            wait_num -= (4 - sz - kz) * 2  # 否则 向听数只减去多余的，即向听数减到为0
    else:  # 如果搭子aa=0，取一张单牌作为将的候选，向听数-1
        wait_num -= 1
        if dzk + dzs12 + dzs13 - (4 - sz - kz) <= 0:  # 向上同理
            wait_num -= (dzk + dzs12 + dzs13) * 2
        else:
            wait_num -= (4 - sz - kz) * 2
        # print tempList

    common_waiting['common_waiting' + str(wait_num)] = 1
    return wait_num