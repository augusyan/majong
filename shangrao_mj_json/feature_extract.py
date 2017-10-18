# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : feature_extract.py
@time : 2017/9/30 10:04
@function : 实现向听数的四种胡牌方式的计算
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
    # return common_waiting
    print(common_waiting)


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
    for i in tile_list:
        # print("the %d has %d in list" % (i, tile_list.count(i)))
        if tile_list.count(i) >= 2:
            wait_num -= 1
    # print(tile_list)
    # wait_types_7['seven_waiting'+str(wait_num)] = 1
    # print(wait_num)
    wait_7couples['seven_waiting'+str(wait_num)] = 1
    print(wait_7couples)
    # return wait_num


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
    print(wait_13lan)


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
    print(wait_19)


def calculate_13(tiles):
    # 计算十三浪的数牌最大向听数
    return max((tiles.count(1)+tiles.count(4)+tiles.count(7)), (tiles.count(1)+tiles.count(4)+tiles.count(8)), \
               (tiles.count(1) + tiles.count(4) + tiles.count(9)), (tiles.count(1)+tiles.count(5)+tiles.count(8)), \
               (tiles.count(1) + tiles.count(5) + tiles.count(9)), (tiles.count(2)+tiles.count(5)+tiles.count(8)), \
               (tiles.count(2) + tiles.count(5) + tiles.count(9)), tiles.count(3)+tiles.count(6)+tiles.count(9))


def station_change():
    # 实现通过actions列表生成不同的hands
    print('hahn')


def num_tile_2_8(tile_list):
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
    for c in tile_list:
        if c & 0x0F > 1 and c & 0x0F < 9:
            ++n
    num_tile['num_tile' + str(n)] = 1
    return num_tile


# 某种花色牌最多的张数
def most_tile_flowerCards(tile_list):
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
    for i in range(len(tile_list) - 1):
        if tile_list[i] & 0xF0 == tile_list[i + 1] & 0xF0:
            n += 1
            # print(n)
            if i == len(tile_list) - 2:
                count[str(tile_list[i] & 0xF0)] = n
        else:
            count[str(tile_list[i] & 0xF0)] = n
            n = 1
    if tile_list[len(tile_list) - 1] & 0xF0 != tile_list[len(tile_list) - 2] & 0xF0:
        count[str(tile_list[len(tile_list) - 1]) & 0xF0] = 1
    temp = 0
    for key in count:
        # print(key, "=", count[key])
        if key != 3 and temp < count[key]:
            temp = count[key]
    most_tile['most_tile' + str(temp)] = 1
    return most_tile


# 各色1到9有无
def num_has1to9(tile_list):  # 1万到9万，一条到9条，一筒到9筒
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
    # print(tile_list)
    for c in tile_list:
        if c & 0xF0 == 0x00:
            has_1to9['num_19_wan_' + str(c & 0x0F)] = 1
        elif c & 0xF0 == 0x10:
            has_1to9['num_19_tiao_' + str(c & 0x0F)] = 1
        elif c & 0xF0 == 0x20:
            has_1to9['num_19_tong_' + str(c & 0x0F)] = 1
    return has_1to9


# 备齐3张以上的相同牌的个数
def num_same_3tile(tile_list):  # 3张以上相同的牌，0到4
    same_3tile = {'same_3tile_0': 0,
                  'same_3tile_1': 0,
                  'same_3tile_2': 0,
                  'same_3tile_3': 0,
                  'same_3tile_4': 0,
                  }
    count = {}
    n = 1
    for i in range(len(tile_list) - 1):
        if tile_list[i] == tile_list[i + 1]:
            n += 1
            # print(n)
            if i == len(tile_list) - 2:
                count[str(tile_list[i])] = n
        else:
            count[str(tile_list[i])] = n
            n = 1
    if tile_list[len(tile_list) - 1] != tile_list[len(tile_list) - 2]:
        count[str(tile_list[len(tile_list) - 1])] = 1
    count_same3 = 0
    for key in count:
        # print(key, "=", count[key])
        if count[key] >= 3:
            count_same3 += 1

    same_3tile['same_3tile_' + str(count_same3)] = 1
    return same_3tile


# 备齐2张以上相同牌的个数
def num_same_2tile(tile_list):  # 2张以上相同的牌，0到6
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
    for i in range(len(tile_list) - 1):
        if tile_list[i] == tile_list[i + 1]:
            n += 1
            # print(n)

            if i == len(tile_list) - 2:
                count[str(tile_list[i])] = n
        else:
            count[str(tile_list[i])] = n
            n = 1
    if tile_list[len(tile_list) - 1] != tile_list[len(tile_list) - 2]:
        count[str(tile_list[len(tile_list) - 1])] = 1
    count_same2 = 0
    for key in count:
        # print(key, "=", count[key])
        if count[key] >= 2:
            count_same2 += 1

    same_2tile['same_3tile_' + str(count_same2)] = 1
    return same_2tile


def num_fulu(actions):
    num_suit = {'suit_0': 0, 'suit_1': 0, 'suit_2': 0, 'suit_4': 0, 'suit_' + str(len(actions)): 1}
    print(num_suit)
    return num_suit

# cards:吃碰杠动作组合


def has_1to9_action(actions):
    has_action = {'suit_num_chow_12': 0,
                  'suit_num_chow_13': 0,
                  'suit_num_chow_23': 0,
                  'suit_num_chow_24': 0,
                  'suit_num_chow_34': 0,
                  'suit_num_chow_35': 0,
                  'suit_num_chow_45': 0,
                  'suit_num_chow_46': 0,
                  'suit_num_chow_56': 0,
                  'suit_num_chow_57': 0,
                  'suit_num_chow_67': 0,
                  'suit_num_chow_68': 0,
                  'suit_num_chow_78': 0,
                  'suit_num_chow_79': 0,
                  'suit_num_chow_89': 0,
                  'suit_num_pung_1': 0,
                  'suit_num_pung_2': 0,
                  'suit_num_pung_3': 0,
                  'suit_num_pung_4': 0,
                  'suit_num_pung_5': 0,
                  'suit_num_pung_6': 0,
                  'suit_num_pung_7': 0,
                  'suit_num_pung_8': 0,
                  'suit_num_pung_9': 0,
                  'suit_num_pung_w': 0,
                  'suit_num_pung_x': 0,
                  'suit_num_pung_y': 0,
                  'suit_num_pung_z': 0,
                  'suit_num_pung_i': 0,
                  'suit_num_pung_j': 0,
                  'suit_num_pung_k': 0,
                  'suit_num_gang_1': 0,
                  'suit_num_gang_2': 0,
                  'suit_num_gang_3': 0,
                  'suit_num_gang_4': 0,
                  'suit_num_gang_5': 0,
                  'suit_num_gang_6': 0,
                  'suit_num_gang_7': 0,
                  'suit_num_gang_8': 0,
                  'suit_num_gang_9': 0,
                  'suit_num_gang_w': 0,
                  'suit_num_gang_x': 0,
                  'suit_num_gang_y': 0,
                  'suit_num_gang_z': 0,
                  'suit_num_gang_i': 0,
                  'suit_num_gang_j': 0,
                  'suit_num_gang_k': 0, }
    for d in actions:
        if (d & 0x0F000) / (16 ** 3) == 3:  # 判断吃

            if ((d & 0x000F0) / 16) == ((d & 0x0000F) + 1) and \
                                    (d & 0x00F00) / (16 ** 2) < 3:
                # 确定操作牌位置及去掉字牌
                # 操作牌在中间
                print('dddddddddddddddddddddddddd')
                has_action['suit_num_chow_' + str(d & 0x0000F)
                           + str((d & 0x0000F) + 2)] = 1
            elif (d & 0x000F0) / 16 == (d & 0x0000F) + 2 and \
                                    (d & 0x00F00) / (16 ** 2) < 3:  # 操作牌在后面
                has_action['suit_num_chow_' + str(d & 0x0000F)
                           + str((d & 0x0000F) + 1)] = 1
            elif (d & 0x000F0) / 16 == d & 0x0000F and \
                                    (d & 0x00F00) / (16 ** 2) < 3:  # 操作牌在前面
                has_action['suit_num_chow_' + str((d & 0x0000F) + 1)
                           + str((d & 0x0000F) + 2)] = 1
        if (d & 0x0F000) / (16 ** 3) == 4:  # 判断碰
            if (d & 0x00F00) / (16 ** 2) < 3:
                has_action['suit_num_pung_' + str((d & 0x000F0) / 16)] = 1
            else:
                if (d & 0x000F0) / 16 == 1 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_pung_w'] = 1
                if (d & 0x000F0) / 16 == 2 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_pung_x'] = 1
                if (d & 0x000F0) / 16 == 3 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_pung_y'] = 1
                if (d & 0x000F0) / 16 == 4 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_pung_z'] = 1
                if (d & 0x000F0) / 16 == 5 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_pung_i'] = 1
                if (d & 0x000F0) / 16 == 6 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_pung_j'] = 1
                if (d & 0x000F0) / 16 == 7 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_pung_k'] = 1
        if (d & 0x0F000) / (16 ** 3) == 5 or (d & 0x0F000) / (16 ** 3) \
                == 6 or (d & 0x0F000) / (16 ** 3) == 7:  # 判断杠
            if (d & 0x00F00) / (16 ** 2) < 3:
                has_action['suit_num_gang_' + str((d & 0x000F0) / 16)] = 1
            else:
                if (d & 0x000F0) / 16 == 1 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_gang_w'] = 1
                if (d & 0x000F0) / 16 == 2 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_gang_x'] = 1
                if (d & 0x000F0) / 16 == 3 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_gang_y'] = 1
                if (d & 0x000F0) / 16 == 4 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_gang_z'] = 1
                if (d & 0x000F0) / 16 == 5 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_gang_i'] = 1
                if (d & 0x000F0) / 16 == 6 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_gang_j'] = 1
                if (d & 0x000F0) / 16 == 7 and (d & 0x00F00) / (16 ** 2) == 3:
                    has_action['suit_num_gang_k'] = 1
    print(has_action)
    return has_action


# 连续二搭+面子候补


def has_2dhb(cards):
    has_2card = {'suit_no19_continue12': 0,
                 'suit_no19_continue23': 0,
                 'suit_no19_continue34': 0,
                 'suit_no19_continue45': 0,
                 'suit_no19_continue56': 0,
                 'suit_no19_continue67': 0,
                 'suit_no19_continue78': 0,
                 'suit_no19_continue89': 0,
                 'suit_no19_same11': 0,
                 'suit_no19_same22': 0,
                 'suit_no19_same33': 0,
                 'suit_no19_same44': 0,
                 'suit_no19_same55': 0,
                 'suit_no19_same66': 0,
                 'suit_no19_same77': 0,
                 'suit_no19_same88': 0,
                 'suit_no19_same99': 0,
                 'suit_no19_chow13': 0,
                 'suit_no19_chow24': 0,
                 'suit_no19_chow35': 0,
                 'suit_no19_chow46': 0,
                 'suit_no19_chow57': 0,
                 'suit_no19_chow68': 0,
                 'suit_no19_chow79': 0, }
    tempList = cards
    # 123,算12，23吗？(如果算，则要先抽走顺子)
    # for i in range(len(cards)-1):
    #        if i+1 <= len(cards)-1:  # 合法牌堆
    #           if cards[i] == cards[i+1]:  # 抽对子
    #                if i == 0:  # 第一张牌不考虑其前面的牌是多少。
    #                    if i+2 <= len(cards)-1:  # 是否存在第三张牌
    #                        if cards[i] != cards[i+2]:
    #                            has_2dhb['suit_no19_same'+str(cards[i] & 0x0F)
    #                                     + str(cards[i] & 0x0F)] = 1
    #                    else:
    #                        has_2dhb['suit_no19_same' + str(cards[i] & 0x0F) +
    #                                 str(cards[i] & 0x0F)] = 1
    #                else:   # 除第一张牌之外的牌，要多考虑是否与前面的牌相同
    #                    if i+2 <= len(cards)-1:
    #                        if cards[i] != cards[i+2] and cards[i] \
    #                           != cards[i-1]:
    #                            has_2dhb['suit_no19_same'+str(cards[i] & 0x0F)
    #                                     + str(cards[i] & 0x0F)] = 1
    #                    elif cards[i] != cards[i-1]:
    #                        has_2dhb['suit_no19_same'+str(cards[i] & 0x0F)
    #                                 + str(cards[i] & 0x0F)] = 1
    #            elif cards[i]+1 == cards[i+1]:  # 判断是否是连续2
    #                has_2dhb['suit_no19_continue'+str(cards[i] & 0x0F)
    #                         + str(cards[i+1] & 0x0F)] = 1
    #            elif cards[i]+2 == cards[i+1]:  # 类似坎张。（135算几次？）
    #                has_2dhb['suit_no19_chow'+str(cards[i] & 0x0F)
    #                         + str(cards[i+1] & 0x0F)] = 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~删除顺子~~~~~~~~~~~~~~~~~~~~~~~~~~
    i = 0
    while i <= len(tempList) - 3:
        if tempList[i] & 0xF0 != 0x30 and tempList[i] + 1 in tempList and \
                                tempList[i] + 2 in tempList:
            card0 = tempList[i]
            card1 = tempList[i] + 1
            card2 = tempList[i] + 2
            tempList.remove(card0)
            tempList.remove(card1)
            tempList.remove(card2)
        else:
            i += 1
    # ~~~~~~~~~~~~~~~~~~~~删除刻子~~~~~~~~~~~~~~~~~~~~~
    j = 0
    while j <= len(tempList) - 3:
        if tempList[j + 1] == tempList[j] and tempList[j + 2] == tempList[j]:

            card = tempList[j]
            tempList.remove(card)
            tempList.remove(card)
            tempList.remove(card)
        else:
            j += 1
    # ~~~~~~~~~~~~~~~~~~~~~判断continue~~~~~~~~~~~~~~~~~~~~~~~~~~~
    k = 0
    while k <= len(tempList) - 2:
        if tempList[k] & 0xF0 != 0x30:

            if tempList[k] + 1 in tempList:
                card0 = tempList[k]
                card1 = tempList[k] + 1
                has_2card['suit_no19_continue' + str(card0 & 0x0F) +
                          str(card1 & 0x0F)] = 1
                tempList.remove(card0)
                tempList.remove(card1)
            else:
                k += 1
    # ~~~~~~~~~~~~~~~~~~~~~判断same~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x = 0
    while x <= len(tempList) - 2:

        if tempList[x + 1] == tempList[x]:
            card = tempList[x]
            has_2card['suit_no19_same' + str(card & 0x0F)
                      + str(card & 0x0F)] = 1
            tempList.remove(card)
            tempList.remove(card)
        else:
            x += 1
    # ~~~~~~~~~~~~~~~~~~~~~判断chow~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    y = 0
    if tempList[y] + 2 in tempList:
        card0 = tempList[y]
        card2 = tempList[y] + 2
        has_2card['suit_no19_chow' + str(card0 & 0x0F) +
                  str(card2 & 0x0F)] = 1
        tempList.remove(card0)
        tempList.remove(card2)
    else:
        y += 1
    return has_2card


def num_fandx(actions, cards):
    num_fxx = {'suit0_waiting_0': 0,
               'suit0_waiting_1': 0,
               'suit0_waiting_2': 0,
               'suit0_waiting_3': 0,
               'suit0_waiting_4': 0,
               'suit0_waiting_5': 0,
               'suit0_waiting_6': 0,
               'suit0_waiting_7': 0,
               'suit0_waiting_8': 0,
               'suit0_waiting_9': 0,
               'suit0_waiting_10': 0,
               'suit0_waiting_11': 0,
               'suit0_waiting_12': 0,
               'suit0_waiting_13': 0,
               'suit0_waiting_14': 0,
               'suit1_waiting_0': 0,
               'suit1_waiting_1': 0,
               'suit1_waiting_2': 0,
               'suit1_waiting_3': 0,
               'suit1_waiting_4': 0,
               'suit1_waiting_5': 0,
               'suit1_waiting_6': 0,
               'suit1_waiting_7': 0,
               'suit1_waiting_8': 0,
               'suit1_waiting_9': 0,
               'suit1_waiting_10': 0,
               'suit1_waiting_11': 0,
               'suit2_waiting_0': 0,
               'suit2_waiting_1': 0,
               'suit2_waiting_2': 0,
               'suit2_waiting_3': 0,
               'suit2_waiting_4': 0,
               'suit2_waiting_5': 0,
               'suit2_waiting_6': 0,
               'suit2_waiting_7': 0,
               'suit2_waiting_8': 0,
               'suit3_waiting_0': 0,
               'suit3_waiting_1': 0,
               'suit3_waiting_2': 0,
               'suit3_waiting_3': 0,
               'suit3_waiting_4': 0,
               'suit3_waiting_5': 0,
               'suit4_waiting_0': 0,
               'suit4_waiting_1': 0,
               }

    tempList = cards
    # print tempList
    wait_num = 14
    act_num = len(actions)
    wait_num = wait_num - act_num * 3
    print(wait_num)
    sz = 0  # 顺子数
    kz = 0  # 刻子数
    dzk = 0  # 搭子 aa
    dzs12 = 0  # 搭子ab
    dzs13 = 0  # 搭子ac
    # 判断顺子数
    i = 0
    while i <= len(tempList) - 3:
        if tempList[i] & 0xF0 != 0x30 and tempList[i] + 1 in tempList and \
                                tempList[i] + 2 in tempList:

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
    print(tempList)

    # 判断刻子数
    j = 0
    while j <= len(tempList) - 3:
        if tempList[j + 1] == tempList[j] and tempList[j + 2] == tempList[j]:

            wait_num -= 3
            kz += 1
            card = tempList[j]
            tempList.remove(card)
            tempList.remove(card)
            tempList.remove(card)
        else:
            j += 1
    print
    tempList

    # 判断搭子aa
    x = 0
    while x <= len(tempList) - 2:

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
    num_fxx['suit' + str(act_num) + '_waiting_' + str(wait_num)] = 1
    return num_fxx

#备齐2或3张的役牌
def suit_tile(cards):
    feature = {
        'suit_2tile_0': 0,
        'suit_2tile_1': 0,
        'suit_2tile_2': 0,
        'suit_2tile_3': 0,
        'suit_2tile_4': 0,
        'suit_2tile_5': 0,
        'suit_2tile_6': 0,
        'suit_2tile_7': 0,
        'suit_3tile_0': 0,
        'suit_3tile_1': 0,
        'suit_3tile_2': 0,
        'suit_3tile_3': 0,
        'suit_3tile_4': 0
    }

    yi = [0,0,0,0,0,0,0]         #役牌持有数
    zi = [0x31,0x32,0x33,0x34,0x35,0x36,0x37]      #字牌数组

    #役牌赋值
    for i in range(len(cards)):
        for j in range(len(zi)):
            if cards[i] == zi[j]:
                yi[j] = yi[j] + 1

    m = 0
    n = 0

    for k in yi:
        if k == 2:
            m = m + 1
        elif k == 3:
            n = n + 1

    feature['suit_2tile_' + str(m)] = 1
    feature['suit_3tile_' + str(n)] = 1

    return feature


#是否鸣19*19向听减少
def suit_19_waitnum(cards,actions):
    feature = {
        'suit_19_waitnum0': 0,
        'suit_19_waitnum1': 0,
        'suit_19_waitnum2': 0,
        'suit_no19_waitnum0': 0,
        'suit_no19_waitnum1': 0,
        'suit_no19_waitnum2': 0
    }

    _19 = 0     #是否鸣19

    #判断19
    for i in actions:
        if (i & 0x0F000)/(16**3) == 3:
            if (i & 0x0000F) == 1 or (i & 0x0000F) == 7:
                _19 = 1

    xiangTingShu = common_waiting(cards)    #获取向听数特征

    #判断向听数
    if xiangTingShu['common_waiting0'] == 1:
        xiangTingDec = 0
    elif xiangTingShu['common_waiting1'] == 1:
        xiangTingDec = 1
    else:
        xiangTingDec = 2

    if _19 == 1:
        feature['suit_19_waitnum' + str(xiangTingDec)] = 1
    elif _19 == 0:
        feature['suit_no19_waitnum' + str(xiangTingDec)] = 1

    return feature


#副露数*字牌持有数
def suit_zi(cards,actions):
    feature = {
        'suit0_x1':0,
        'suit0_x2':0,
        'suit0_x3':0,
        'suit0_x4':0,
        'suit0_y1':0,
        'suit0_y2':0,
        'suit0_y3':0,
        'suit0_y4':0,
        'suit0_w1':0,
        'suit0_w2':0,
        'suit0_w3':0,
        'suit0_w4':0,
        'suit0_z1':0,
        'suit0_z2':0,
        'suit0_z3':0,
        'suit0_z4':0,
        'suit1_x1':0,
        'suit1_x2':0,
        'suit1_x3':0,
        'suit1_x4':0,
        'suit1_y1':0,
        'suit1_y2':0,
        'suit1_y3':0,
        'suit1_y4':0,
        'suit1_w1':0,
        'suit1_w2':0,
        'suit1_w3':0,
        'suit1_w4':0,
        'suit1_z1':0,
        'suit1_z2':0,
        'suit1_z3':0,
        'suit1_z4':0,
        'suit2_x1':0,
        'suit2_x2':0,
        'suit2_x3':0,
        'suit2_x4':0,
        'suit2_y1':0,
        'suit2_y2':0,
        'suit2_y3':0,
        'suit2_y4':0,
        'suit2_w1':0,
        'suit2_w2':0,
        'suit2_w3':0,
        'suit2_w4':0,
        'suit2_z1':0,
        'suit2_z2':0,
        'suit2_z3':0,
        'suit2_z4':0,
        'suit3_x1':0,
        'suit3_x2':0,
        'suit3_x3':0,
        'suit3_x4':0,
        'suit3_y1':0,
        'suit3_y2':0,
        'suit3_y3':0,
        'suit3_y4':0,
        'suit3_w1':0,
        'suit3_w2':0,
        'suit3_w3':0,
        'suit3_w4':0,
        'suit3_z1':0,
        'suit3_z2':0,
        'suit3_z3':0,
        'suit3_z4':0,
        'suit0_i1':0,
        'suit0_i2':0,
        'suit0_i3':0,
        'suit0_i4':0,
        'suit0_j1':0,
        'suit0_j2':0,
        'suit0_j3':0,
        'suit0_j4':0,
        'suit0_k1':0,
        'suit0_k2':0,
        'suit0_k3':0,
        'suit0_k4':0,
        'suit1_i1':0,
        'suit1_i2':0,
        'suit1_i3':0,
        'suit1_i4':0,
        'suit1_j1':0,
        'suit1_j2':0,
        'suit1_j3':0,
        'suit1_j4':0,
        'suit1_k1':0,
        'suit1_k2':0,
        'suit1_k3':0,
        'suit1_k4':0,
        'suit2_i1':0,
        'suit2_i2':0,
        'suit2_i3':0,
        'suit2_i4':0,
        'suit2_j1':0,
        'suit2_j2':0,
        'suit2_j3':0,
        'suit2_j4':0,
        'suit2_k1':0,
        'suit2_k2':0,
        'suit2_k3':0,
        'suit2_k4':0,
        'suit3_i1':0,
        'suit3_i2':0,
        'suit3_i3':0,
        'suit3_i4':0,
        'suit3_j1':0,
        'suit3_j2':0,
        'suit3_j3':0,
        'suit3_j4':0,
        'suit3_k1':0,
        'suit3_k2':0,
        'suit3_k3':0,
        'suit3_k4':0
    }

    xi = ['w','x','y','z','i','j','k']    #字牌所代表字母
    yi = [0,0,0,0,0,0,0]    #役牌持有数
    zi = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]     #字牌16进制数

    #役牌赋值
    for i in range(len(cards)):
        for j in range(len(zi)):
            if cards[i] == zi[j]:
                yi[j] = yi[j] + 1

    fuLu = 0    #副露数
    for k in actions:
        act = (k & 0x0F000) / (16 ** 3)
        if act == 3 or act == 4 or act == 5 or act == 6:
            fuLu = fuLu + 1

    for a in xi:
        for b in yi:
            feature['suit' + str(fuLu) + '_' + a + str(b)] = 1

    return feature


#副露数*向听数*向听是否减少
def suit_waiting(cards,actions):
    festure = {
        'suit0_waiting0_dec':0,
        'suit0_waiting1_dec':0,
        'suit0_waiting2_dec':0,
        'suit0_waiting3_dec':0,
        'suit1_waiting0_dec':0,
        'suit1_waiting1_dec':0,
        'suit1_waiting2_dec':0,
        'suit1_waiting3_dec':0,
        'suit2_waiting0_dec':0,
        'suit2_waiting1_dec':0,
        'suit2_waiting2_dec':0,
        'suit2_waiting3_dec':0,
        'suit3_waiting0_dec':0,
        'suit3_waiting1_dec':0,
        'suit3_waiting2_dec':0,
        'suit3_waiting3_dec':0,
        'suit4_waiting0_dec':0,
        'suit4_waiting1_dec':0,
        'suit4_waiting2_dec':0,
        'suit4_waiting3_dec':0,
        'suit0_waiting0_stay':0,
        'suit0_waiting1_stay':0,
        'suit0_waiting2_stay':0,
        'suit0_waiting3_stay':0,
        'suit1_waiting0_stay':0,
        'suit1_waiting1_stay':0,
        'suit1_waiting2_stay':0,
        'suit1_waiting3_stay':0,
        'suit2_waiting0_stay':0,
        'suit2_waiting1_stay':0,
        'suit2_waiting2_stay':0,
        'suit2_waiting3_stay':0,
        'suit3_waiting0_stay':0,
        'suit3_waiting1_stay':0,
        'suit3_waiting2_stay':0,
        'suit3_waiting3_stay':0,
        'suit4_waiting0_stay':0,
        'suit4_waiting1_stay':0,
        'suit4_waiting2_stay':0,
        'suit4_waiting3_stay':0
    }

    #获取副露数
    n = 0
    for i in actions:
        for j in [3,4,5,6]:
            if (i & 0x0F000)/(16**3) == j:
                n = n + 1
    fuLu = n

    #获取向听数
    xiangTing = 0
    xiangTingShu = common_waiting(cards)
    for k in range(len(xiangTingShu)):
        if xiangTingShu[k] == 1:
            xiangTing = k

    dec = 0     #向听数减少

    if dec == 1:
        festure['suit'+str(fuLu)+'_waiting'+str(xiangTing)+'_dec'] = 1
    elif dec == 0:
        festure['suit' + str(fuLu) + '_waiting' + str(xiangTing) + '_stay'] = 1

    return festure


#向听数
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
            print(tempList[i], "i")
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
    print(tempList)

    # 判断刻子数
    j = 0
    while j <= len(tempList) - 3:
        if tempList[j + 1] == tempList[j] and tempList[j + 2] == tempList[j]:
            print(tempList[j], "j")
            wait_num -= 3
            kz += 1
            card = tempList[j]
            tempList.remove(card)
            tempList.remove(card)
            tempList.remove(card)
        else:
            j += 1
    print(tempList)

    # 判断搭子aa
    x = 0
    while x <= len(tempList) - 2:
        print(tempList[x], "x")
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
            print(tempList[k], "k")
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
    print(tempList)

    common_waiting['common_waiting' + str(wait_num)] = 1
    return common_waiting

hands_test = [0x02, 0x03, 0x03, 0x05, 0x06, 0x09, 0x03, 0x14, 0x16, 0x28, 0x32, 0x34, 0x36]
actions_test = [0x21020, 0x22320, 0x21310, 0x22360, 0x21070, 0x22310, 0x21160, 0x22340, 0x24160, \
                0x22280, 0x21340, 0x22340, 0x21020, 0x22090, 0x21350, 0x22350, 0x21140, 0x22130, \
                0x21290, 0x22290, 0x26020, 0x21180, 0x22180, 0x21180, 0x22180, 0x21350, 0x22350, \
                0x21030, 0x28000]

"""
actions_test = ['2G2m', '2dx', '2Gw', '2dj', '2G7m', '2dw', '2G6s',\
                '2dz', '2N6s', '2d8p', '2Gz', '2dz', '2G2m', '2d9m',\
                '2Gi', '2di', '2G4s', '2d3s', '2G9p', '2d9p', '2k2m', \
                '2G8s', '2d8s', '2G8s', '2d8s', '2Gi', '2di', '2G3m', '2A']
"""

wait_types_comm(tile_list=hands_test)
wait_types_7(tile_list=hands_test)
wait_types_13(tile_list=hands_test)
wait_types_19(tile_list=hands_test)
print(num_tile_2_8(tile_list=hands_test))
print(most_tile_flowerCards(tile_list=hands_test))
print(num_has1to9(tile_list=hands_test))
print(num_same_3tile(tile_list=hands_test))
print(num_same_2tile(tile_list=hands_test))

# print(hands_test)
# print(actions_test)
