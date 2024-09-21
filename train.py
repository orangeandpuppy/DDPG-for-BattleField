import os
import numpy as np
from tqdm import tqdm
from net.ddpg import DDPG
from config import config as cfg
from config import MAX_FIGHTER
from utils import create_env_from_info, plot_learning_curve, create_directory, create_info, parse_game_data_from_file


if __name__ == "__main__":
    # model
    n_actions = 12*MAX_FIGHTER
    n_states = 10*MAX_FIGHTER
    agent = MDDPG(cfg, n_states, n_actions)
    create_directory(path=cfg['out_dir'], sub_paths=['/Actor', '/Target_actor', '/Critic', '/Target_critic'])

    print(f"n_action is {n_actions}, n_state is {n_states}")

    # load
    if cfg['train_load']:
        agent.load_models(episode=cfg['episode'])
    print(f"Store model in {cfg['out_dir']}")

    reward_history = []
    avg_reward_history = []
    first = True
    best_reward = 0

    # train
    for episode in range(cfg['epochs']):
        done = False
        total_reward = 0
        if cfg['train_name']:
            info_dict = parse_game_data_from_file(cfg['train_name'])
        else:
            info_dict = create_info()
        env = create_env_from_info(info_dict)
        agent.n_fighter = len(env.fighters_info)
        state = env.reset()
        begin = np.random.randint(0, cfg['memory_capacity']//2)
        for i in tqdm(range(cfg['memory_capacity']), desc='Episode: {}'.format(episode + 1)):
            action = agent.choose_action(state, train=True)
            state_, reward, done, info_dict = env.step(action)
            agent.store_transition(state, action, reward, state_, done)
            if agent.f_memory_counter + agent.z_memory_counter >= agent.memory_capacity and i >= begin:
                agent.learn()
            total_reward += reward
            state = state_
            if done:
                break
        if first or best_reward < total_reward:
            agent.save_models('best')
            best_reward = total_reward
            first = False

        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-5:])
        avg_reward_history.append(avg_reward)
        print('Ep: {} Reward: {:.1f} AvgReward: {:.1f} sum_value:{}:'.format(episode + 1, total_reward, avg_reward, env.sum_value))

        if (episode + 1) % cfg['epoch_save'] == 0:
            agent.save_models(episode + 1)

    episodes = [i+1 for i in range(cfg['epochs'])]
    if cfg['train_name']:
        plot_learning_curve(episodes, reward_history, title='BattleField', ylabel='reward',
                            figure_file=os.path.join(cfg['out_dir'], 'reward.png'))
    else:
        plot_learning_curve(episodes, avg_reward_history, title='BattleField', ylabel='reward',
                            figure_file=os.path.join(cfg['out_dir'], 'reward.png'))
