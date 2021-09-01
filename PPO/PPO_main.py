#!/usr/bin/python
# -- coding:utf-8 --
'''
        ====================================================================================================
         .----------------.  .----------------.  .----------------.  .----------------.  .----------------.
        | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
        | |     ______   | || |      __      | || |    _______   | || |     _____    | || |      __      | |
        | |   .' ___  |  | || |     /  \     | || |   /  ___  |  | || |    |_   _|   | || |     /  \     | |
        | |  / .'   \_|  | || |    / /\ \    | || |  |  (__ \_|  | || |      | |     | || |    / /\ \    | |
        | |  | |         | || |   / ____ \   | || |   '.___`-.   | || |      | |     | || |   / ____ \   | |
        | |  \ `.___.'\  | || | _/ /    \ \_ | || |  |`\____) |  | || |     _| |_    | || | _/ /    \ \_ | |
        | |   `._____.'  | || ||____|  |____|| || |  |_______.'  | || |    |_____|   | || ||____|  |____|| |
        | |              | || |              | || |              | || |              | || |              | |
        | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
         '----------------'  '----------------'  '----------------'  '----------------'  '----------------'


                 .----------------.    .----------------.    .----------------.    .----------------.
                | .--------------. |  | .--------------. |  | .--------------. |  | .--------------. |
                | |     _____    | |  | |    _______   | |  | |    _______   | |  | |     ______   | |
                | |    |_   _|   | |  | |   /  ___  |  | |  | |   /  ___  |  | |  | |   .' ___  |  | |
                | |      | |     | |  | |  |  (__ \_|  | |  | |  |  (__ \_|  | |  | |  / .'   \_|  | |
                | |      | |     | |  | |   '.___`-.   | |  | |   '.___`-.   | |  | |  | |         | |
                | |     _| |_    | |  | |  |`\____) |  | |  | |  |`\____) |  | |  | |  \ `.___.'\  | |
                | |    |_____|   | |  | |  |_______.'  | |  | |  |_______.'  | |  | |   `._____.'  | |
                | |              | |  | |              | |  | |              | |  | |              | |
                | '--------------' |  | '--------------' |  | '--------------' |  | '--------------' |
                 '----------------'    '----------------'    '----------------'    '----------------'
        ====================================================================================================
          Current Task : PPO_ZWS | User : Wensheng Zhang
        ====================================================================================================
'''
import argparse
import torch
import gym
import numpy as np
import os
import random
from tensorboardX import SummaryWriter
from PPO_Agent import Agent
from PPO_env import ENV as environment
from itertools import count

def main(env, buffer_size, batch_size, horizon_steps, update_steps, render, episode=10000,
         lamda=0.95, gamma=0.99, lr=0.0001, tau=1e-3, run_type=1, clip=0.2, max_grad_norm=0.5, seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(env, device, buffer_size, batch_size, horizon_steps, update_steps, lamda, gamma, lr, tau, run_type, clip, max_grad_norm, seed)
    for i in range(int(episode)):
        state = env.reset()
        score = 0
        for t in count():
            if render: env.render()
            action, action_pro = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(t, state, action, (reward + 8) / 8, done, next_state, action_pro)
            state = next_state
            score += reward
            if done:
                agent.writer.add_scalar('Steptime/steptime', t, global_step=i)
                print('I_ep {} ，train {} times, scores {}'.format(i+1, t+1, score), end=' ')
                if score >= -200:
                # if score >= 195:
                    print("=========>  SUCCESS")
                else:
                    print("   ")

                break

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='PPO')
    parser.add_argument('--env_name', type=str, default='Pendulum-v0')
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--buffer_size", type=int, default=1e6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--horizon_steps", type=int, default=256)
    parser.add_argument("--update_steps", type=int, default=10)
    parser.add_argument('--render', action='store_true', default=False, help='render the environment')
    parser.add_argument("--episode", type=int, default=1e6)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.9)')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument("--run_type", type=int, default=1)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='interval between training status logs (default: 10)')
    args = parser.parse_args()
    args.env_name = "Pendulum-v0"
    # args.env_name = 'CartPole-v0'
    # args.run_type = 0
    env = environment(args.env_name)
    main(env, args.buffer_size, args.batch_size, args.horizon_steps, args.update_steps, args.render, args.episode,
         args.lamda, args.gamma, args.lr, args.tau, args.run_type, args.clip, args.max_grad_norm, args.seed)