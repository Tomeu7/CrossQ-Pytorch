import torch
import torch.nn as nn
import time
from src.methods.utils import initialise, ReplayMemory, get_parameters_by_name
from typing import Tuple
import numpy as np

class Agent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config_agent: dict,
                 device: int,
                 print_update_every: int) -> None:
        """
        Agent class
        state_dim: number of state features
        action_dim: number of action features
        config_agent: config for agent
        device: device used for training
        """

        self.device = "cuda:" + str(device)
        self.config_agent = config_agent
        # Initializing SAC/CrossQ
        self.target_entropy, self.log_alpha, self.alpha_optim, \
            self.policy, self.critic, self.critic_target, \
            self.policy_optim, self.critic_optim =\
            initialise(state_dim=state_dim,
                       action_dim=action_dim,
                       device=device,
                       agent_type=config_agent['agent_type'],
                       crossQstyle=config_agent['crossqstyle'],
                       num_layers_actor=config_agent['num_layers_actor'],
                       num_layers_critic=config_agent['num_layers_critic'],
                       hidden_dim_critic=config_agent['hidden_dim_critic'],
                       hidden_dim_actor=config_agent['hidden_dim_actor'],
                       lr=config_agent['lr'],
                       use_batch_norm_critic=config_agent['use_batch_norm_critic'],
                       use_batch_norm_policy=config_agent['use_batch_norm_policy'],
                       beta1=config_agent['beta1'],
                       beta2=config_agent['beta2'],
                       beta1_alpha=config_agent['beta1_alpha'],
                       beta2_alpha=config_agent['beta2_alpha'],
                       lr_alpha=config_agent['lr_alpha'],
                       automatic_entropy_tuning=config_agent['automatic_entropy_tuning'],
                       alpha=config_agent['alpha'],
                       entropy_factor=config_agent['entropy_factor'],
                       activation_policy=config_agent['activation_policy'],
                       activation_critic=config_agent['activation_critic'],
                       bn_momentum=config_agent['bn_momentum'],
                       bn_mode=config_agent['bn_mode'],
                       remove_target_network=config_agent['remove_target_network']
                       )
        # for bn
        if not self.config_agent['remove_target_network'] and not self.config_agent['crossqstyle']:
            self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
            self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        else:
            self.batch_norm_stats = None
            self.batch_norm_stats_target = None
        self.alpha = config_agent['alpha']

        # Replay buffer
        self.replay_buffer = ReplayMemory(capacity=config_agent['replay_capacity'])
        # Counters
        self.num_updates = 0
        self.print_update_every = print_update_every
        self.print_counter = 0
        # Loss MSE
        self.mse_loss = nn.MSELoss()

        print("----------------------------------------------")
        print(self.critic)
        print(self.policy)
        print("Policy to eval")
        self.policy.eval()
        print("Critic to eval")
        self.critic.eval()
        if self.critic_target is not None:
            print("Critic target to eval - it should always be in evaluation mode")
            self.critic_target.eval()
        print("----------------------------------------------")

    @torch.no_grad()
    def get_tensors_from_memory(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = \
            self.replay_buffer.sample(batch_size=self.config_agent['batch_size'])
        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device).unsqueeze(1)
        mask_batch = mask_batch.to(self.device).unsqueeze(1)
        if self.num_updates == 0:
            print("Shapes replay buffer sample:", state_batch.shape, action_batch.shape, reward_batch.shape, next_state_batch.shape, mask_batch.shape)
        return state_batch, action_batch, reward_batch, next_state_batch, mask_batch

    ###################################################################################################################

    def calculate_q_loss(self, qf1: torch.tensor, qf2: torch.tensor, next_q_value: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Calculates loss with Soft Bellman equation
        :param qf1: Q_1(s,a)
        :param qf2: Q_2(s,a)
        :param next_q_value: Bellman error
        :return: float total loss, float q1 loss, float q2 loss
        """
        q1_loss = self.mse_loss(qf1, next_q_value)
        q2_loss = self.mse_loss(qf2, next_q_value)
        q_loss = q1_loss + q2_loss
        return q_loss, q1_loss, q2_loss

    def update_replay_buffer(self, s, a, r, s_next, done):
        self.replay_buffer.push(s, a, r, s_next, done)
    
    def update(self):
        raise NotImplementedError

    def select_action(self, s: np.ndarray, evaluation: bool):
        raise NotImplementedError

    def update_actor(self, state_batch: torch.tensor):
        raise NotImplementedError

    def calculate_policy_loss(self, log_pi: torch.tensor, qf1_pi: torch.tensor, qf2_pi: torch.tensor):
        raise NotImplementedError

    def update_critic(self,
                      state_batch: torch.tensor,
                      action_batch: torch.tensor,
                      reward_batch: torch.tensor,
                      next_state_batch: torch.tensor,
                      mask_batch: torch.tensor,
                      time_to_print: bool
                      ):
        pass

    def train(self) -> dict:
        """
        Trains agent
        """
        statistics = None
        if len(self.replay_buffer) > self.config_agent['batch_size']:
            
            time_start_iter = time.time()

            statistics = self.update()

            self.num_updates += 1

            if statistics is not None:
                time_update = time.time() - time_start_iter
                statistics['time_update'] = time_update
                additional_info = f"Updater metrics: send policy weights, number of updates: {self.num_updates} - iter_time {time_update:.4f} - Len replay {len(self.replay_buffer)}"
                dynamic_info = " - ".join([f"{key}: {value:.2f}" for key, value in statistics.items()])
                print(f"{additional_info} - {dynamic_info}")
        return statistics

    def save_policy(self, file_name='policy_model_weights.pth'):
        torch.save(self.policy.state_dict(), file_name)

    def load_policy(self, file_name='policy_model_weights.pth'):
        self.policy.load_state_dict(torch.load(file_name))

    def reset_optimizer(self):
        from torch.optim.adam import Adam
        self.policy_optim = Adam(self.policy.parameters(),
                             lr=self.lr)

