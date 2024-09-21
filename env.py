import gym
from gym import spaces
import numpy as np
import math
from config import MAX_FIGHTER
from config import config as cfg


class BattleFieldEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, map_layout, red_bases_info, blue_bases_info, fighters_info):
        super(BattleFieldEnv, self).__init__()

        self.map_layout = [list(row) for row in map_layout]  # 地图布局
        self.red_bases = [{'position': pos, 'defense': defense, 'value': value, 'is_destroyed': False} for pos, defense, value in red_bases_info]
        self.blue_bases = [{'position': pos, 'fuel': fuel, 'missiles': missiles} for pos, fuel, missiles in blue_bases_info]
        self.fighters_info = [{'id': f_id, 'position': pos, 'max_fuel': fuel, 'max_missiles': missiles, 'fuel': 0, 'missiles': 0} for f_id, pos, fuel, missiles in fighters_info]
        self.blue_bases_positions = dict((tuple(self.blue_bases[i]['position']), i) for i in range(len(self.blue_bases)))
        self.red_bases_positions = dict((tuple(self.red_bases[i]['position']), i) for i in range(len(self.red_bases)))
        # 按fighter['position']计数
        self.fighters_positions = dict()
        for i in range(len(self.fighters_info)):
            if tuple(self.fighters_info[i]['position']) in self.fighters_positions:
                self.fighters_positions[tuple(self.fighters_info[i]['position'])] += 1
            else:
                self.fighters_positions[tuple(self.fighters_info[i]['position'])] = 1

        # self.action_space = spaces.Discrete(12)
        # self.observation_space = spaces.Box(low=0, high=2000, shape=(10, 389), dtype=np.float)

        self.init_map_layout = self.map_layout
        self.init_red_bases_info = self.red_bases
        self.init_blue_bases_info = self.blue_bases
        self.init_fighters_info = self.fighters_info

        self.n = len(self.map_layout)
        self.m = len(self.map_layout[0])
        self.sight = cfg['sight']
        self.direction_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        self.rdirection = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3, (0, 0): 4}
        self.pre_action = np.zeros((MAX_FIGHTER, 12), dtype=np.float32)

        self.sum_value = 0
        for base in self.red_bases:
            self.sum_value += base['value']

        self.mode = 'train'
        self.pre_state = None
        self.maxnum = 200

    def _handle_move_action(self, i, direction, file=None):
        if direction == 4:
            if self.mode == 'train':
                return -2, False  # or -1?
            elif self.mode == 'test':
                # 如果是测试模式，飞机必须移动，以免陷入无限循环
                if not self.pre_state[i][4] == self.maxnum:
                    direction = self.rdirection[(np.sign(self.pre_state[i][4]), np.sign(self.pre_state[i][5]))]
                    steps = abs(self.pre_state[i][4]) + abs(self.pre_state[i][5])
                    if (steps > self.pre_state[i][8] or self.pre_state[i][9] < self.pre_state[i][6]) and (not self.pre_state[i][0] == self.maxnum):
                        direction = self.rdirection[(np.sign(self.pre_state[i][0]), np.sign(self.pre_state[i][1]))]
                elif not self.pre_state[i][0] == self.maxnum:
                    direction = self.rdirection[(np.sign(self.pre_state[i][0]), np.sign(self.pre_state[i][1]))]
                else:
                    direction = np.random.randint(0, 4)

        # 当前战斗机的位置
        x, y = self.fighters_info[i]['position']

        # 计算新位置
        dx, dy = self.direction_deltas[direction]
        new_x, new_y = x + dx, y + dy

        if self.mode == 'test':
            possible_directions = [0, 1, 2, 3]

            while True:
                direction = np.random.choice(possible_directions)
                dx, dy = self.direction_deltas[direction]
                new_x, new_y = x + dx, y + dy

                if 0 <= new_x < self.n and 0 <= new_y < self.m and self.map_layout[new_x][new_y] != '#':
                    break
                else:
                    possible_directions.remove(direction)

        if 0 <= new_x < self.n and 0 <= new_y < self.m and self.map_layout[new_x][new_y] != '#' and self.fighters_info[i]['fuel'] >= 1:
            self.fighters_positions[(x, y)] -= 1
            if self.fighters_positions[(x, y)] == 0:
                del self.fighters_positions[(x, y)]
            # 更新fighters_info中的位置信息
            self.fighters_info[i]['position'] = (new_x, new_y)
            self.fighters_info[i]['fuel'] -= 1
            if (new_x, new_y) in self.fighters_positions:
                self.fighters_positions[(new_x, new_y)] += 1
            else:
                self.fighters_positions[(new_x, new_y)] = 1

            if file:
                file.write(f'move {self.fighters_info[i]["id"]} {direction}\n')
                print(f'move {self.fighters_info[i]["id"]} {direction}\n')
            reward = -1
            # 增加接近敌方基地的奖励
            if direction == 0:
                e_i = -self.sight
                s_i = 0
                e_j = -self.sight
                s_j = self.sight + 1
            elif direction == 1:
                e_i = 0
                s_i = self.sight
                e_j = -self.sight
                s_j = self.sight + 1
            elif direction == 2:
                e_i = -self.sight
                s_i = self.sight + 1
                e_j = -self.sight
                s_j = 0
            else:
                e_i = -self.sight
                s_i = self.sight + 1
                e_j = 0
                s_j = self.sight + 1
            for i in range(e_i, s_i):
                for j in range(e_j, s_j):
                    if (x + i, y + j) in self.red_bases_positions:
                        reward += self.red_bases[self.red_bases_positions[(x + i, y + j)]]['value']*0.001
            return reward, False

        return -5, False

    def _handle_attack_action(self, i, attack_direction, missile_count: int = 0, p: float = 1.0, file=None):
        delta = self.direction_deltas[attack_direction]
        target_position = (self.fighters_info[i]['position'][0] + delta[0], self.fighters_info[i]['position'][1] + delta[1])

        if target_position in self.red_bases_positions:
            target_base_id = self.red_bases_positions[target_position]
        else:
            return 0, False, 0  # 无有效攻击目标，实际投放弹药数量为0  or return 0？免得这一部分太容易学习成只分配0个missile了

        if file:
            file.write(f'attack {self.fighters_info[i]["id"]} {attack_direction} {missile_count}\n')
            print(f'attack {self.fighters_info[i]["id"]} {attack_direction} {missile_count}\n')

        more_missile = 0
        if self.red_bases[target_base_id]['defense'] <= missile_count:
            more_missile = missile_count - self.red_bases[target_base_id]['defense']
        self.fighters_info[i]['missiles'] -= missile_count
        self.red_bases[target_base_id]['defense'] -= missile_count

        if self.red_bases[target_base_id]['defense'] <= 0:
            self.red_bases[target_base_id]['is_destroyed'] = True
            del self.red_bases_positions[target_position]
            x, y = self.red_bases[target_base_id]['position']
            self.map_layout[x][y] = '.'
            # 基地被摧毁，返回正奖励和实际投放的弹药数量
            return self.red_bases[target_base_id]['value'] - more_missile + missile_count, bool(
                len(self.red_bases_positions) == 0), missile_count
        else:
            return missile_count, False, missile_count  # 成功攻击，但基地未被摧毁，返回正奖励和实际投放的弹药数量

    def _handle_refuel_action(self, i, refuel_count: float = 1.0, file=None):
        current_pos = self.fighters_info[i]['position']
        if current_pos in self.blue_bases_positions:
            base_id = self.blue_bases_positions[current_pos]
        else:
            return 0, False, 0

        fuel_add = math.floor(self.blue_bases[base_id]['fuel'] * refuel_count)
        actual_fuel_added = min(fuel_add, self.fighters_info[i]['max_fuel'] - self.fighters_info[i]['fuel'])

        self.fighters_info[i]['fuel'] += actual_fuel_added
        self.blue_bases[base_id]['fuel'] -= actual_fuel_added

        if file and actual_fuel_added != 0:
            file.write(f'fuel {self.fighters_info[i]["id"]} {actual_fuel_added}\n')
            print(f'fuel {self.fighters_info[i]["id"]} {actual_fuel_added}\n')

        return 0, False, actual_fuel_added

    def _handle_reload_action(self, i, reload_count: float = 1.0, file=None):
        current_pos = self.fighters_info[i]['position']
        if current_pos in self.blue_bases_positions:
            base_id = self.blue_bases_positions[current_pos]
        else:
            return 0, False, 0

        missile_add = math.floor(self.blue_bases[base_id]['missiles'] * reload_count)
        actual_missiles_added = min(missile_add,
                                    self.fighters_info[i]['max_missiles'] - self.fighters_info[i]['missiles'])

        self.fighters_info[i]['missiles'] += actual_missiles_added
        self.blue_bases[base_id]['missiles'] -= actual_missiles_added

        if file and actual_missiles_added != 0:
            file.write(f'missile {self.fighters_info[i]["id"]} {actual_missiles_added}\n')
            print(f'missile {self.fighters_info[i]["id"]} {actual_missiles_added}\n')

        return 0, False, actual_missiles_added

    def step(self, action, file=None):
        # action: [MAX_FIGHTER, 12]
        done = False
        reward = 0
        counts = []
        refuel = []
        reload = []

        for i in range(min(len(self.fighters_info), MAX_FIGHTER)):
            act = action[i]

            # refuel
            p_refuel = act[10]
            rf_reward, rf_done, rf_count = self._handle_refuel_action(i, p_refuel, file)

            # reload
            p_reload = act[11]
            rl_reward, rl_done, rl_count = self._handle_reload_action(i, p_reload, file)

            # attack
            attack = act[5:10]
            a_reward = 0
            a_done = False
            count = 0
            initial_missiles = self.fighters_info[i]['missiles']
            for j in range(4):
                aa_reward, aa_done, ccount = self._handle_attack_action(i, j, math.floor(initial_missiles*attack[j]), attack[j], file)
                a_reward += aa_reward
                a_done = a_done or aa_done
                count += ccount

            # move
            move = act[:5]
            direction = np.argmax(move)
            m_reward, m_done = self._handle_move_action(i, direction, file)

            reward += m_reward + a_reward + rf_reward + rl_reward
            done = m_done or a_done or rf_done or rl_done or done
            counts.append(count)
            refuel.append(rf_count)
            reload.append(rl_count)

        self.pre_action = action
        return [self._get_observation(), reward, done, {'counts': counts, 'refuel': refuel, 'reload': reload}]

    def reset(self):
        # 重置环境状态到初始状态
        # 这里需要添加重置逻辑，例如重置战斗机和基地状态
        self.map_layout = self.init_map_layout
        self.fighters_info = self.init_fighters_info
        self.red_bases = self.init_red_bases_info
        self.blue_bases = self.init_blue_bases_info

        self.blue_bases_positions = dict((tuple(self.blue_bases[i]['position']), i) for i in range(len(self.blue_bases)))
        self.red_bases_positions = dict((tuple(self.red_bases[i]['position']), i) for i in range(len(self.red_bases)))
        # 按fighter['position']计数
        self.fighters_positions = dict()
        for i in range(len(self.fighters_info)):
            if tuple(self.fighters_info[i]['position']) in self.fighters_positions:
                self.fighters_positions[tuple(self.fighters_info[i]['position'])] += 1
            else:
                self.fighters_positions[tuple(self.fighters_info[i]['position'])] = 1

        self.pre_action = np.zeros((MAX_FIGHTER, 12), dtype=np.float32)
        return self._get_observation()

    def _get_observation(self):
        # 返回当前环境的观测
        # 最近的一个己方基地的相对x坐标，相对y坐标，基地剩余燃油，基地剩余导弹 (4)
        # 最近的一个敌方基地的相对x坐标，相对y坐标，基地防御值，基地价值      (4)
        # 剩余燃油，剩余导弹                                          (2)
        observations = np.zeros((MAX_FIGHTER, 10), dtype=np.float32)
        for i in range(len(self.fighters_info)):
            fighter = self.fighters_info[i]
            observation = np.zeros(10, dtype=np.float32)
            # 把observation[0]设为一个很大的数，如果没有找到基地，就可以知道
            observation[0] = self.maxnum
            observation[4] = self.maxnum

            get_blue = False
            get_red = False
            for j in range(0, self.sight + 1):
                for k in range(4):
                    x, y = fighter['position']
                    x += self.direction_deltas[k][0] * j
                    y += self.direction_deltas[k][1] * j

                    if (x, y) in self.blue_bases_positions and not get_blue:
                        base = self.blue_bases[self.blue_bases_positions[(x, y)]]
                        observation[0] = x - fighter['position'][0]
                        observation[1] = y - fighter['position'][1]
                        observation[2] = base['fuel']
                        observation[3] = base['missiles']
                        get_blue = True

                    if (x, y) in self.red_bases_positions and not get_red:
                        base = self.red_bases[self.red_bases_positions[(x, y)]]
                        observation[4] = x - fighter['position'][0]
                        observation[5] = y - fighter['position'][1]
                        observation[6] = base['defense']
                        observation[7] = base['value']
                        get_red = True

                    if get_blue and get_red:
                        break

            observation[8] = fighter['fuel']
            observation[9] = fighter['missiles']
            observations[i] = observation

        return observations

