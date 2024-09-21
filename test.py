import os
import time
import numpy as np
from config import config as cfg
from config import MAX_FIGHTER
from net.ddpg import DDPG
from utils import parse_game_data_from_file, create_env_from_info

if __name__ == "__main__":
    # 取test_name的文件名（不含后缀
    test_name = os.path.splitext(os.path.basename(cfg['test_name']))[0]
    # 取test_name除文件名外的上层目录
    test_dir = os.path.dirname(cfg['test_name'])
    # 取load_dir的文件名
    load_name = os.path.basename(cfg['load_dir'])

    file_name = test_name + '_' + load_name + '_' + str(cfg['episode']) + '.out'
    file_name = os.path.join(test_dir, file_name)
    file = open(file_name, 'w')

    # model
    n_actions = 12*MAX_FIGHTER
    n_states = 10*MAX_FIGHTER
    agent = MDDPG(cfg, n_states, n_actions)
    agent.load_models(episode=cfg['episode'])

    # data
    info_dict = parse_game_data_from_file(cfg['test_name'])
    env = create_env_from_info(info_dict)
    env.mode = 'test'

    # test
    start_time = time.time()
    state = env.reset()
    done = False
    count = 0
    while not done:
        if env.pre_state is not None and np.array_equal(env.pre_state, state):
            print("Equal!")
            add_noise = True
        else:
            add_noise = False
        action = agent.choose_action(state, train=add_noise)
        env.pre_state = state
        state_, reward, done, info_dict = env.step(action, file)
        state = state_
        end_time = time.time()
        file.write('OK\n')
        print("OK\n")
        count += 1
        if count == 15010 or end_time - start_time > 120:
            break

    file.close()
