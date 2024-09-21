import matplotlib.pyplot as plt
from env import BattleFieldEnv
import os
import random
import torch.nn as nn
from config import MAX_N, MAX_M, MAX_G, MAX_C, MAX_D, MAX_V, MAX_FIGHTER, MAX_FG, MAX_FC


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def display_game_data(game_data):
    n, m = game_data['map_size']
    map_layout = game_data['map_layout']
    blue_bases = game_data['blue_bases']
    red_bases = game_data['red_bases']
    fighters = game_data['fighters']

    fig, ax = plt.subplots(figsize=(m, n))
    plt.xlim(-0.5, m - 0.5)
    plt.ylim(-0.5, n - 0.5)

    # 绘制地图布局
    for y, row in enumerate(map_layout):
        for x, cell in enumerate(row):
            if cell == '.':
                ax.plot(x, n - 1 - y, marker='s', color='w', markersize=30)
            elif cell == '*':
                ax.plot(x, n - 1 - y, marker='*', color='b', markersize=10)  # 蓝方基地
            elif cell == '#':
                ax.plot(x, n - 1 - y, marker='p', color='r', markersize=10)  # 红方基地

    # 标记蓝方基地
    for base in blue_bases:
        x, y = base['position']
        ax.plot(x, n - 1 - y, marker='o', color='lightblue', markersize=15)

    # 标记红方基地
    for base in red_bases:
        x, y = base['position']
        ax.plot(x, n - 1 - y, marker='o', color='salmon', markersize=15)

    # 标记战斗机
    for fighter in fighters:
        x, y = fighter[:2]
        ax.plot(x, n - 1 - y, marker='^', color='green', markersize=10)

    plt.gca().invert_yaxis()
    plt.axis('on')
    ax.set_xticks(range(m))
    ax.set_yticks(range(n))
    ax.grid(which='both')

    plt.show()


def create_directory(path: str, sub_paths: list):
    for sub_path in sub_paths:
        if not os.path.exists(path + sub_path):
            os.makedirs(path + sub_path, exist_ok=True)


def create_env_from_info(info_dict):
    map_layout = info_dict['map_layout']
    blue_bases_info = [(base['position'], base['attributes'][0], base['attributes'][1]) for base in
                       info_dict['blue_bases']]
    red_bases_info = [(base['position'], base['attributes'][2], base['attributes'][3]) for base in
                      info_dict['red_bases']]
    fighters_info = [(idx, tuple(fighter[:2]), fighter[2], fighter[3]) for idx, fighter in
                     enumerate(info_dict['fighters'])]
    env = BattleFieldEnv(
        map_layout=map_layout,
        red_bases_info=red_bases_info,
        blue_bases_info=blue_bases_info,
        fighters_info=fighters_info
    )
    return env


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, color='r', linestyle='-')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.savefig(figure_file)
    plt.show()


def parse_game_data_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.read().strip().split('\n')

        # n: 行数, m: 列数
        n, m = map(int, lines[0].split())
        # 地图布局
        map_layout = lines[1: n + 1]
        offset = n + 1

        # 蓝方基地数量
        blue_base_count = int(lines[offset])
        blue_bases = []
        for i in range(offset + 1, offset + 1 + 2 * blue_base_count, 2):
            if i + 1 >= len(lines):  # 检查是否超出索引范围
                raise RuntimeError("错误：基地信息不完整。")
            position = tuple(map(int, lines[i].split()))  # 基地位置
            attributes = list(map(int, lines[i + 1].split()))  # 基地属性
            blue_bases.append({'position': position, 'attributes': attributes})

        offset += 2 * blue_base_count + 1
        # 红方基地数量
        red_base_count = int(lines[offset])
        red_bases = []
        for i in range(offset + 1, offset + 1 + 2 * red_base_count, 2):
            if i + 1 >= len(lines):  # 检查是否超出索引范围
                raise RuntimeError("错误：基地信息不完整。")
            position = tuple(map(int, lines[i].split()))  # 基地位置
            attributes = list(map(int, lines[i + 1].split()))  # 基地属性
            red_bases.append({'position': position, 'attributes': attributes})

        offset += 2 * red_base_count + 1
        if offset >= len(lines):  # 检查战斗机数量行是否存在
            raise RuntimeError("错误：缺少战斗机数量信息。")
        fighter_count = int(lines[offset])  # 战斗机数量
        fighters = []
        for i in range(offset + 1, offset + 1 + fighter_count):
            if i >= len(lines):  # 检查是否超出索引范围
                raise RuntimeError("错误：战斗机信息不完整。")
            fighters.append(list(map(int, lines[i].split())))  # 战斗机属性

        return {
            'map_size': (n, m),
            'map_layout': map_layout,
            'blue_bases': blue_bases,
            'red_bases': red_bases,
            'fighters': fighters
        }


def create_info():
    # 1<=n,m<=200
    n = random.randint(2, MAX_N)
    m = random.randint(2, MAX_M)
    # 我方基地
    blue_base_count = random.randint(1, n * m // 2 - 1)
    blue_bases = []
    for i in range(blue_base_count):
        h = random.randint(0, n - 1)
        w = random.randint(0, m - 1)
        g = random.randint(0, MAX_G)
        c = random.randint(0, MAX_C)
        d = random.randint(1, MAX_D)
        v = random.randint(1, MAX_V)
        blue_bases.append({'position': (h, w), 'attributes': [g, c, d, v]})

    # 敌方基地
    red_base_count = random.randint(1, n * m // 2 - blue_base_count)
    red_bases = []
    for i in range(red_base_count):
        h = random.randint(0, n - 1)
        w = random.randint(0, m - 1)
        g = random.randint(0, MAX_G)
        c = random.randint(0, MAX_C)
        d = random.randint(1, MAX_D)
        v = random.randint(1, MAX_V)
        red_bases.append({'position': (h, w), 'attributes': [g, c, d, v]})

    # 我方战斗机
    fighter_count = random.randint(1, min(MAX_FIGHTER, blue_base_count))
    fighter_id = []
    # 在0~blue_base_count-1中随机选择fighter_count个不重复的基地
    for i in range(fighter_count):
        while True:
            idx = random.randint(0, blue_base_count - 1)
            if idx not in fighter_id:
                fighter_id.append(idx)
                break

    fighters = []
    for i in range(fighter_count):
        h = blue_bases[fighter_id[i]]['position'][0]
        w = blue_bases[fighter_id[i]]['position'][1]
        g = random.randint(1, MAX_FG)
        c = random.randint(1, MAX_FC)
        fighters.append([h, w, g, c])

    map_layout = [''.join(['.' for _ in range(m)]) for _ in range(n)]
    for base in blue_bases:
        h, w = base['position']
        map_layout[h] = map_layout[h][:w] + '*' + map_layout[h][w + 1:]
    for base in red_bases:
        h, w = base['position']
        map_layout[h] = map_layout[h][:w] + '#' + map_layout[h][w + 1:]

    info = {
        'map_size': (n, m),
        'map_layout': map_layout,
        'blue_bases': blue_bases,
        'red_bases': red_bases,
        'fighters': fighters
    }
    return info


if __name__ == "__main__":
    info_dict = parse_game_data_from_file('data/testcase1.in')
    env = create_env_from_info(info_dict)
    env.reset()