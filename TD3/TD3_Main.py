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
import gym
import time
import torch
from itertools import count
from TD3_Agent import Agent
from Arguments import arguments

def Main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env_name = "Pendulum-v0"  # 'CartPole-v0'
    env = gym.make(env_name)

    if env_name == "Pendulum-v0":
        args.action_type = 1
        args.actor_state_dim = args.critic_state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.shape[0]
    elif env_name == 'CartPole-v0':
        args.action_type = 0
        args.actor_state_dim = args.critic_state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.n

    our_agent = Agent(args, device)

    for eps in count():
        """环境初始化"""

        state = env.reset()
        scores = 0

        """运行"""
        for i in count():

            our_action = our_agent.act(state)

            next_state, reward, done, _ = env.step(our_action)

            our_agent.step(i, state, next_state, our_action, reward, done)

            state = next_state

            scores += reward

            if done == True:
                end_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                print("Episode: ", eps, "  End_time:  ", end_time, "  Scores:  ", round(scores, 2), end=' ')
                # if scores >= -200:
                if scores >= 195:
                    print("=========>  SUCCESS")
                else:
                    print("   ")

                break


if __name__ == "__main__":
    args = arguments()
    Main(args)