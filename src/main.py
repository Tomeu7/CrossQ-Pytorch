import gymnasium as gym
import argparse
import json
from src.methods.sac import SAC
from src.methods.utils import ReplayMemory
import os
import pandas as pd
import torch
import numpy as np
import random

def update_config(args_, config_):
    for arg in vars(args_):
        value = getattr(args_, arg)
        if value is not None:
            if arg in ["crossqstyle", "use_batch_norm_critic", "use_batch_norm_policy", "remove_target_network"]:
                config_[arg] = True if value == "true" else False
            else:
                config_[arg] = value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--environment_name", type=str, required=True)
    parser.add_argument("--num_total_steps", default=100000)
    parser.add_argument("--print_update_every", default=1000, type=int)
    parser.add_argument("--print_reward_every", default=1000, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", type=int, default=1234)
    # Not required args
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_alpha", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--update_critic_to_policy_ratio", type=int, default=None)
    parser.add_argument("--num_layers_actor", type=int, default=None)
    parser.add_argument("--num_layers_critic", type=int, default=None)
    parser.add_argument("--hidden_dim_critic", type=int, default=None)
    parser.add_argument("--hidden_dim_actor", type=int, default=None)
    parser.add_argument("--crossqstyle", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--use_batch_norm_critic", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--use_batch_norm_policy", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--beta1_alpha", type=float, default=None)
    parser.add_argument("--beta2_alpha", type=float, default=None)
    parser.add_argument("--activation_critic", type=str, default=None)
    parser.add_argument("--activation_actor", type=str, default=None)
    parser.add_argument("--bn_momentum", type=float, default=None)
    parser.add_argument("--bn_mode", type=str, default=None)
    parser.add_argument("--remove_target_network", type=str, choices=["true", "false"], default=None)
    args = parser.parse_args()


    # Results
    path_save = "results/" + args.experiment_name + "/"
    if not os.path.exists(path_save):
        os.mkdir(path_save)

    config_file_path = 'src/config.json'
    with open(config_file_path, 'r') as config_file:
        config_agent = json.load(config_file)
    update_config(args, config_agent)

    print("----------------")
    print(config_agent)
    print("Creating environment: " + args.environment_name)
    env = gym.make(args.environment_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print("Num. states: {}, Num. actions: {} action space high {} action space low {}".format(state_dim, action_dim, env.action_space.high, env.action_space.low))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    agent = SAC(state_dim=state_dim,
                  action_dim=action_dim,
                  config_agent=config_agent,
                  device=args.device,
                  print_update_every=args.print_update_every)

    statistics_keys = None
    total_step, num_episode, r_every_checkpoint = 0, 0, 0
    list_total_episode_reward, list_episode_reward, list_losses = [], [], []
    while total_step < args.num_total_steps:
        s, _ = env.reset(seed=args.seed+num_episode)
        r_episode, step, done = 0, 0, False
        while not done:
            a = agent.select_action(s, evaluation=False)

            s_next, r, terminated, truncated, _ = env.step(a)

            done = terminated or truncated

            agent.update_replay_buffer(s, a, r, s_next, done)

            statistics_loss = agent.train()

            s = s_next.copy()
            total_step += 1
            step += 1
            r_episode += r
            r_every_checkpoint += r
            if total_step % args.print_reward_every == 0:
                print("Total step {} Episode step {} Reward {}".format(total_step, step, r))
                list_episode_reward.append([total_step, step, r, r_every_checkpoint])
                r_every_checkpoint = 0
            if statistics_loss is not None:
                statistics_keys = ["num_updates"] + list(statistics_loss.keys())
                print("Losses saved", total_step)
                additional_metrics = [agent.num_updates]
                this_list_losses = additional_metrics + [value for _, value in statistics_loss.items()]
                list_losses.append(this_list_losses)
        list_total_episode_reward.append([total_step, step, r_episode])
        print("Num episode {} Total step {} Episode step {} Total reward {}".format(num_episode, total_step, step, r_episode))
        num_episode += 1
        # Saving results after every episode into dataframes
        df_total_episode_reward = pd.DataFrame(list_total_episode_reward, columns=['Total Step', 'Episode Step', 'Total Reward'])
        df_episode_reward = pd.DataFrame(list_episode_reward, columns=['Total Step', 'Episode Step', 'Reward', 'Reward every 1000'])
        df_losses = pd.DataFrame(list_losses, columns=statistics_keys)
        
        df_total_episode_reward.to_csv(path_save + 'total_episode_reward.csv', index=False)
        df_episode_reward.to_csv(path_save + 'episode_reward.csv', index=False)
        df_losses.to_csv(path_save + 'losses.csv', index=False)

    # Save config
    with open(path_save + "config_experiment.json", 'w') as config_file:
        json.dump(config_agent, config_file, indent=4)
    # Save models
    agent.save_policy(path_save + "policy.pth")




