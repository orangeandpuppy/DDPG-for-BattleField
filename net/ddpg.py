import torch
import math
import numpy as np
import torch.nn as nn
from net.actor_network import ActorNetwork
from net.critic_network import CriticNetwork
from config import MAX_FIGHTER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG:
    def __init__(self, cfg, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action
        self.gamma = cfg['gamma']
        self.tau = cfg['tau']
        self.action_noise = cfg['action_noise']
        self.checkpoint_dir = cfg['out_dir']
        self.loadpoint_dir = cfg['load_dir']
        self.positive_sample_proportion = cfg['positive_sample_proportion']

        self.actor = ActorNetwork(cfg['alpha'], n_state, n_action)
        self.target_actor = ActorNetwork(cfg['alpha'], n_state, n_action)
        self.critic = CriticNetwork(cfg['beta'], n_state, n_action)
        self.target_critic = CriticNetwork(cfg['beta'], n_state, n_action)

        self.memory_capacity = cfg['memory_capacity']
        # s, a, r, s_, done
        self.f_memory_counter = 0
        self.f_state_memory = np.zeros((self.memory_capacity, n_state))
        self.f_action_memory = np.zeros((self.memory_capacity, n_action))
        self.f_reward_memory = np.zeros((self.memory_capacity, 1))
        self.f_next_state_memory = np.zeros((self.memory_capacity, n_state))
        self.f_done_memory = np.zeros((self.memory_capacity, 1))

        self.z_memory_counter = 0
        self.z_state_memory = np.zeros((self.memory_capacity, n_state))
        self.z_action_memory = np.zeros((self.memory_capacity, n_action))
        self.z_reward_memory = np.zeros((self.memory_capacity, 1))
        self.z_next_state_memory = np.zeros((self.memory_capacity, n_state))
        self.z_done_memory = np.zeros((self.memory_capacity, 1))

        self.batch_size = cfg['batch_size']
        self.loss_func = nn.MSELoss()

        self.n_action = n_action
        self.n_state = n_state
        self.n_fighter = 0

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic_params, target_critic_params in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)

    def store_transition(self, s, a, r, s_, done):
        if r <= 0:
            index = self.f_memory_counter % self.memory_capacity
            self.f_state_memory[index, :] = s.flatten()
            self.f_action_memory[index, :] = a.flatten()
            self.f_reward_memory[index, :] = r
            self.f_next_state_memory[index, :] = s_.flatten()
            self.f_done_memory[index, :] = done
            self.f_memory_counter += 1
        else:
            index = self.z_memory_counter % self.memory_capacity
            self.z_state_memory[index, :] = s.flatten()
            self.z_action_memory[index, :] = a.flatten()
            self.z_reward_memory[index, :] = r
            self.z_next_state_memory[index, :] = s_.flatten()
            self.z_done_memory[index, :] = done
            self.z_memory_counter += 1

    def choose_action(self, observation, train=True):
        self.actor.eval()
        # action: [MAX_FIGHTER, 12]
        action = torch.zeros((MAX_FIGHTER, 12), dtype=torch.float32).to(device)
        # 把每个智能体的观测状态分开，并转换为tensor
        for i in range(MAX_FIGHTER):
            state = torch.tensor(np.array([observation[i]]), dtype=torch.float).to(device)
            # action: [1, 12] -> [12,]
            act = self.actor.forward(state).squeeze()
            action[i, :] = act

        if train:
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise, size=action.shape), dtype=torch.float).to(device)
            action = action + noise
            move = action[..., :5]
            move = torch.softmax(move, dim=-1)
            attack = action[..., 5:10]
            attack = torch.softmax(attack, dim=-1)
            fuels = action[..., 10]
            fuels = torch.sigmoid(fuels)
            fuels = fuels.unsqueeze(-1)
            missiles = action[..., 11]
            missiles = torch.sigmoid(missiles)
            missiles = missiles.unsqueeze(-1)
            action = torch.cat([move, attack, fuels, missiles], dim=-1)
        self.actor.train()

        return action.detach().cpu().numpy()

    def learn(self):
        z_b = min(self.z_memory_counter, math.floor(self.batch_size*self.positive_sample_proportion))
        f_b = self.batch_size - z_b
        if f_b > self.f_memory_counter:
            f_b = self.f_memory_counter
            z_b = self.batch_size - f_b
        z_sample_index = np.random.choice(min(self.memory_capacity, self.z_memory_counter), z_b)
        z_states = torch.FloatTensor(self.z_state_memory[z_sample_index].reshape([z_b, MAX_FIGHTER, self.n_state // MAX_FIGHTER]))
        z_actions = torch.FloatTensor(self.z_action_memory[z_sample_index].reshape([z_b, MAX_FIGHTER, self.n_action // MAX_FIGHTER]))
        z_reward = torch.FloatTensor(self.z_reward_memory[z_sample_index])
        z_states_ = torch.FloatTensor(self.z_next_state_memory[z_sample_index].reshape([z_b, MAX_FIGHTER, self.n_state // MAX_FIGHTER]))
        z_terminals = torch.BoolTensor(self.z_done_memory[z_sample_index])

        f_sample_index = np.random.choice(min(self.memory_capacity, self.f_memory_counter), f_b)
        f_states = torch.FloatTensor(self.f_state_memory[f_sample_index].reshape([f_b, MAX_FIGHTER, self.n_state // MAX_FIGHTER]))
        f_actions = torch.FloatTensor(self.f_action_memory[f_sample_index].reshape([f_b, MAX_FIGHTER, self.n_action // MAX_FIGHTER]))
        f_reward = torch.FloatTensor(self.f_reward_memory[f_sample_index])
        f_states_ = torch.FloatTensor(self.f_next_state_memory[f_sample_index].reshape([f_b, MAX_FIGHTER, self.n_state // MAX_FIGHTER]))
        f_terminals = torch.BoolTensor(self.f_done_memory[f_sample_index])

        states = torch.cat([z_states, f_states], dim=0)
        actions = torch.cat([z_actions, f_actions], dim=0)
        reward = torch.cat([z_reward, f_reward], dim=0)
        states_ = torch.cat([z_states_, f_states_], dim=0)
        terminals = torch.cat([z_terminals, f_terminals], dim=0)
        states_tensor = states.to(device)
        # actions: [b, MAX_FIGHTER, 12]
        actions_tensor = actions.to(device)
        # reward: [b, 1]
        rewards_tensor = reward.to(device)
        next_states_tensor = states_.to(device)
        # terminals: [b, 1]
        terminals_tensor = terminals.to(device)

        with torch.no_grad():
            # next_actions_tensor: [b, MAX_FIGHTER, 12]
            next_actions_tensor = torch.zeros((self.batch_size, MAX_FIGHTER, 12), dtype=torch.float32).to(device)
            for i in range(MAX_FIGHTER):
                next_actions_tensor[:, i, :] = self.target_actor.forward(next_states_tensor[:, i, :])
            # q_ :[b,]
            q_ = self.target_critic.forward(next_states_tensor, next_actions_tensor)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_
        q = self.critic.forward(states_tensor, actions_tensor)

        critic_loss = self.loss_func(q, target.detach())
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # new_actions_tensor: [b, MAX_FIGHTER, 12]
        new_actions_tensor = torch.zeros((self.batch_size, MAX_FIGHTER, 12), dtype=torch.float32).to(device)
        for i in range(MAX_FIGHTER):
            new_actions_tensor[:, i, :] = self.actor.forward(states_tensor[:, i, :])
        actor_loss = -torch.mean(self.critic(states_tensor, new_actions_tensor))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def save_models(self, episode):
        print(f"Saving model to {self.checkpoint_dir} {episode}")
        self.actor.save_checkpoint(self.checkpoint_dir + '/Actor/DDPG_actor_{}.pth'.format(episode))
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          '/Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        self.critic.save_checkpoint(self.checkpoint_dir + '/Critic/DDPG_critic_{}'.format(episode))
        self.target_critic.save_checkpoint(self.checkpoint_dir +
                                           '/Target_critic/DDPG_target_critic_{}'.format(episode))
        print('Saving model successfully!')

    def load_models(self, episode: int = 50):
        print(f"Loading model from {self.loadpoint_dir} {episode}")
        self.actor.load_checkpoint(self.loadpoint_dir + '/Actor/DDPG_actor_{}.pth'.format(episode))
        self.target_actor.load_checkpoint(self.loadpoint_dir +
                                          '/Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        self.critic.load_checkpoint(self.loadpoint_dir + '/Critic/DDPG_critic_{}'.format(episode))
        self.target_critic.load_checkpoint(self.loadpoint_dir +
                                           '/Target_critic/DDPG_target_critic_{}'.format(episode))
        print('Loading model successfully!')