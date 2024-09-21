import time
import os

MAX_N = 200
MAX_M = 200
MAX_G = 1000
MAX_C = 2000
MAX_D = 10
MAX_V = 200
MAX_FIGHTER = 5
MAX_FG = 500
MAX_FC = 1000

config = {
    'net': 'DDPG',
    'batch_size': 32,
    'memory_capacity': 1000,
    'tau': 0.005,
    'action_noise': 0.2,
    'gamma': 0.99,
    'sight': 10,

    'alpha': 0.003,
    'beta': 0.003,

    'epochs': 40,
    'epoch_save': 10,
    'positive_sample_proportion': 0.5,
    'load_dir': 'result/20240527-200048',
    'episode': 'best',

    'train_name': 'data/testcase2.in',
    'train_load': True,

    'test_name': 'data/testcase2.in',

    'RESULT_PATH': 'result'
}

exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
config['out_dir'] = os.path.join(config['RESULT_PATH'], exp_id)

