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

class arguments():
    def __init__(self):
        """fc网络层数"""
        self.fclayer_num = 3

        """状态空间"""
        self.state_space = 3

        """动作空间"""
        self.action_space = 1

        """中间层网络节点数"""
        self.net_dim = 256

        """SAC训练方式：v1，v2"""
        self.learn_type = 2

        """动作离散/连续：0离散，1连续"""
        self.action_type = 1

        """动作采样模式: 0：sample采样， 1：取argmax"""
        self.action_mode = 0

        """动作最值"""
        self.max_action = 2

        """lr: 学习率"""
        self.lr = 2.5e-4

        """lr: 学习率衰减步长"""
        self.lr_anneal_step = 1e5

        """lr: 学习率衰减率"""
        self.lr_anneal = 0.95

        """buffer_size"""
        self.buffer_size = 1e6

        """batch_size"""
        self.batch_size = 128

        """gamma: """
        self.gamma = 0.99

        """GAE: lamda系数"""
        self.lamda = 0.95

        """update_each_episode"""
        self.update_steps = 10

        """训练步长"""
        self.horizon_steps = 512  # 512, 2018

        """Clip"""
        self.clip = 0.15

        """refresh_grad_clip"""
        self.max_grad_norm = 0.5

        """熵损失系数"""
        self.entropy_factor = 0.01

        """SAC中熵系数计算方式"""
        self.alpha_auto_entropy = 1

        """1, GAE_delta_n = r_n + gamma * V_n+1 - V_n      0, GAE_delta_n = Q_n - V_n"""
        self.delta_type = 1

        """1,GAE按照轨迹序列无限长近似结果: GAE_t = delta_n + gamma * lamda * GAE_r + 1    
           0,GAE按照轨迹序列有限长推导结果，k为当前step离trajectory最后一步step的距离: GAE_t = delta_n + gamma * lamda * GAE_r + 1 * [(1 - lamda^k)/(1 - lamda^(k + 1))]"""
        self.infinite = 1

        """GAE归一化 1,归一化   0,不归一化"""
        self.norm = 0

        """target网络更新权重"""
        self.tau = 0.01

        """随机种子: elegant：1943， tianshou：1626"""
        self.seed = 1943