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
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network():
    def __init__(self, args):
        self.actor = Actor(args).to(torch.device("cpu"))
        self.critic = Value(args).to(torch.device("cpu"))
        self.paras = Paras(self.actor, self.critic)

    def load(self, path, side, time):
        self.actor.load_network(path, side, time)
        self.critic.load_network(path, side, time)

    def save(self, path, side, time, episode):
        self.actor.save_network(path, side, time, episode)
        self.critic.save_network(path, side, time, episode)

    def get_paras(self, paras):
        self.paras = paras
        self.actor.load_state_dict(paras.actor_paras)
        self.critic.load_state_dict(paras.critic_paras)

    def set_paras(self):
        self.paras = Paras(self.actor, self.critic)

class Paras():
    def __init__(self, actor, critic):
        self.actor_paras = actor.state_dict()
        self.critic_paras = critic.state_dict()

class Actor(nn.Module):
    def __init__(self, args, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        """参数设置"""
        self.args = args
        self.action_type = args.action_type
        self.a_std = torch.zeros(1, 1) + 0.5
        self.min_val = 1e-7
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # 加入torch.manual_seed随机种子这一项后网络无法通过pipe进行传递，因此去掉
        # self.seed = torch.manual_seed(seed)

        """首层网络建立及初始化"""
        self.fc_start = nn.Linear(args.actor_state_dim, args.net_dim)
        layer_init(self.fc_start)

        """中间层网络建立及初始化"""
        for layer in range(args.fclayer_num):
            setattr(self, 'fc' + str(layer+1), nn.Linear(args.net_dim, args.net_dim))
            layer_init(eval("self." + 'fc' + str(layer+1)))

        """输出层网络建立及初始化"""

        setattr(self, 'agent', nn.Linear(args.net_dim, args.action_dim))
        layer_init(eval("self." + 'agent'))
        setattr(self, 'amean', nn.Linear(args.net_dim, args.action_dim))
        layer_init(eval("self." + 'amean'))
        setattr(self, 'std', nn.Linear(args.net_dim, args.action_dim))
        layer_init(eval("self." + 'std'))

    def forward(self, state, valid=None):
        """首层及中间层推演"""
        x = F.relu(self.fc_start(state))
        for layer in range(self.args.fclayer_num):
            x = F.relu(getattr(self, 'fc' + str(layer+1))(x))

        """输出层推演"""
        if self.action_type == 1:

            amean = getattr(self, 'amean')(x)
            a_std = getattr(self, 'std')(x)
            log_std = torch.clamp(a_std, self.log_std_min, self.log_std_max)

            return amean, log_std

        else:
            action_prob = F.softmax(getattr(self, 'agent')(x), dim=-1) + self.min_val
            if valid != None:
                action_prob = action_prob * valid
                action_prob = action_prob/torch.sum(action_prob)

            return action_prob

    def save_network(self, path, side, time, episode):
        if os.path.exists(path + side + "actor/" + time) is False:
            os.mkdir(path + side + "actor/" + time)
        print("Saving episode: {} actor-network...".format(episode))
        torch.save(self.state_dict(), path + side + "actor/" + time + "/episode_{}".format(episode) + ".pth")

    def load_network(self, path, side, time):
        if os.path.exists(path) is False:
            raise ValueError("The folder does not exist!")
        elif os.listdir(path) is None:
            raise ValueError("Could not find the param file in folder!")
        else:
            print("Loading actor-network...")
            get_dir = os.listdir(path + self.args.the_side[side] + "actor/" + time)
            self.load_state_dict(torch.load(path+self.args.the_side[side]+"actor/" + time + get_dir[0]))


class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()
        """参数设置"""
        self.args = args
        self.action_type = args.action_type
        self.a_std = torch.zeros(1, 1) + 0.5
        self.min_val = 1e-7

        # 加入torch.manual_seed随机种子这一项后网络无法通过pipe进行传递，因此去掉
        # self.seed = torch.manual_seed(seed)

        """首层网络建立及初始化"""
        self.fc_start = nn.Linear(args.critic_state_dim, args.net_dim)
        layer_init(self.fc_start)

        """中间层网络建立及初始化"""
        for layer in range(args.fclayer_num):
            setattr(self, 'fc' + str(layer+1), nn.Linear(args.net_dim, args.net_dim))
            layer_init(eval("self." + 'fc' + str(layer+1)))

        """输出层网络建立及初始化"""
        self.fc_end = nn.Linear(args.net_dim, 1)
        layer_init(self.fc_end)

    def forward(self, state):
        """推演"""
        x = F.relu(self.fc_start(state))
        for layer in range(self.args.fclayer_num):
            x = F.relu(getattr(self, 'fc' + str(layer+1))(x))

        return self.fc_end(x)

    def save_network(self, path, side, time, episode):
        if os.path.exists(path + side + "critic/" + time) is False:
            os.mkdir(path + side + "critic/" + time)
        print("Saving episode: {} critic-network...".format(episode))
        torch.save(self.state_dict(), path + side + "critic/" + time + "/episode_{}".format(episode) + ".pth")

    def load_network(self, path, side, time):
        if os.path.exists(path) is False:
            raise ValueError("The folder does not exist!")
        elif os.listdir(path) is None:
            raise ValueError("Could not find the param file in folder!")
        else:
            print("Loading critic-network...")
            get_dir = os.listdir(path+self.args.the_side[side]+"critic/" + time)
            self.load_state_dict(torch.load(path+self.args.the_side[side]+"critic/" + time + get_dir[0]))

class Q_net(nn.Module):
    def __init__(self, args):
        super(Q_net, self).__init__()
        """参数设置"""
        self.args = args
        self.action_type = args.action_type
        self.a_std = torch.zeros(1, 1) + 0.5
        self.min_val = 1e-7

        # 加入torch.manual_seed随机种子这一项后网络无法通过pipe进行传递，因此去掉
        # self.seed = torch.manual_seed(seed)

        """首层网络建立及初始化"""
        if args.action_type == 1:
            self.fc_start = nn.Linear(args.critic_state_dim + args.action_dim, args.net_dim)
        else:
            self.fc_start = nn.Linear(args.critic_state_dim, args.net_dim)
        layer_init(self.fc_start)

        """中间层网络建立及初始化"""
        for layer in range(args.fclayer_num):
            setattr(self, 'fc' + str(layer + 1), nn.Linear(args.net_dim, args.net_dim))
            layer_init(eval("self." + 'fc' + str(layer + 1)))

        """输出层网络建立及初始化"""
        if args.action_type == 1:
            self.fc_end = nn.Linear(args.net_dim, 1)
        else:
            self.fc_end = nn.Linear(args.net_dim, args.action_dim)
        layer_init(self.fc_end)

    def forward(self, state, action=None):
        """推演"""
        if self.action_type == 1:
            x = F.relu(self.fc_start(torch.cat((state, action), -1)))
            for layer in range(self.args.fclayer_num):
                x = F.relu(getattr(self, 'fc' + str(layer+1))(x))
        else:
            x = F.relu(self.fc_start(state))
            for layer in range(self.args.fclayer_num):
                x = F.relu(getattr(self, 'fc' + str(layer+1))(x))

        return self.fc_end(x)

    def save_network(self, path, side, time, episode):
        if os.path.exists(path + side + "critic/" + time) is False:
            os.mkdir(path + side + "critic/" + time)
        print("Saving episode: {} critic-network...".format(episode))
        torch.save(self.state_dict(), path + side + "critic/" + time + "/episode_{}".format(episode) + ".pth")

    def load_network(self, path, side, time):
        if os.path.exists(path) is False:
            raise ValueError("The folder does not exist!")
        elif os.listdir(path) is None:
            raise ValueError("Could not find the param file in folder!")
        else:
            print("Loading critic-network...")
            get_dir = os.listdir(path+self.args.the_side[side]+"critic/" + time)
            self.load_state_dict(torch.load(path+self.args.the_side[side]+"critic/" + time + get_dir[0]))

def layer_init(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)    # Tensor正交初始化
    torch.nn.init.constant_(layer.bias, bias_const) # 偏置常数初始化