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
          Current Task :  | User : Wensheng Zhang
        ====================================================================================================  
'''
import argparse
import torch
import gym
import numpy as np
import os
import random
from tensorboardX import SummaryWriter
from ppo_continous.PPO_Agent import Agent as PPO_Agent
from ddpg.DDPG_Agent import Agent as DDPG_Agent
from main_TZ import EachEpoch
from itertools import count
import matplotlib.pyplot as plt
from study_policy import Label

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='DDPG')
    parser.add_argument('--env_name', type=str, default='Pendulum-v0')
    parser.add_argument("--state_dim", type=int, default=6)
    parser.add_argument("--action_dim", type=int, default=18)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--buffer_size", type=int, default=1e6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--horizon_steps", type=int, default=256)
    parser.add_argument("--update_steps", type=int, default=10)
    parser.add_argument('--render', action='store_true', default=False, help='render the environment')
    parser.add_argument("--episode", type=int, default=1e6)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.9)')
    parser.add_argument('--lr', type=float, default=1e-3)  # 1e-3
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument("--run_type", type=int, default=1)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--entropy_factor', type=float, default=0.01)
    parser.add_argument("--delta_type", type=int, default=0)
    parser.add_argument("--infinite", type=int, default=0)
    parser.add_argument("--norm", type=int, default=0)
    parser.add_argument("--max_search_step", type=int, default=2000)
    parser.add_argument("--max_train_step", type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # p_Agent = PPO_Agent(args.state_dim, args.action_dim, device, args.buffer_size, args.batch_size, args.horizon_steps,
    #             args.update_steps, args.lamda, args.gamma, args.lr, args.tau, args.run_type, args.clip,
    #             args.max_grad_norm, args.entropy_factor, args.delta_type, args.infinite, args.norm, args.seed)
    d_Agent = DDPG_Agent(args.state_dim, args.action_dim, device, args.buffer_size, args.batch_size, args.lr, args.gamma, args.tau)

    ee = EachEpoch(False)

    label = Label(args.state_dim, args.action_dim, args.lr*500, args.gamma, args.seed, device)

    action_max = np.array([415, 360, 360])
    action_min = np.array([0, 0, 0])

    input_state = np.array([1, 1, 1, 1, 1, 1])

    max_score = -9999
    max_action = np.random.rand(args.action_dim)

    for episode in count():  # range(args.max_search_step)

        action = d_Agent.act(input_state, False)

        ratio_list = np.split((np.tanh(action)+1)/2, int(args.action_dim/3))
        send_action = action_min + (action_max - action_min) * ratio_list
        current_score = ee.run(send_action)
        ee.reset()

        d_Agent.step(input_state, action, current_score/100)

        if current_score > max_score:
            max_action = action
            max_send_action = send_action
            max_score = current_score
            print("\rCurrent_best_score: {} \nCurrent_best_action:{}\n".format(max_score, max_send_action))
        print('\rSearch_episode:{}    Score:{} \naction:{}\n'.format(episode, current_score, send_action), end=" ")
 
        if (episode+1) % args.max_search_step == 0:
    # max_score = 615
    # max_action = np.array([1.0457387, - 0.02182625,  0.91288334,  1.2316165, - 1.996634, 1.6281404, - 1.9038373, - 0.42540294,  3.2475345, - 0.5344968,   0.06618121,  2.1394134, 1.0309268, - 2.3849967,   0.05007295,  0.9351754,   0.8130758,   0.2168962])
            label.train(input_state, max_action, max_score, args.max_train_step)
