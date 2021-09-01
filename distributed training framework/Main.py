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
import time
import torch
import threading
from copy import deepcopy
from itertools import count
from collections import namedtuple, deque
from tensorboardX import SummaryWriter
from multiprocessing import Process, Pipe, Queue
from ..Algorithm.Argumens import arguments
from ..Algorithm.ppo.PPO_Agent import Agent
from ..Algorithm.ppo.PPO_Model import Network
from ..Algorithm.ppo.env_shell import ENV
from ..Algorithm.ppo.PPO_Model import Actor, Critic
from ..GeneralEnv.my_import import my_import

def Average(array):
    avg = 0.0
    n = len(array)
    for num in array:
        avg += 1.0*num/n
    return avg

class Main_thread(threading.Thread):
    def __init__(self, args, conn_array, side, device, thread_id):
        threading.Thread.__init__(self)
        self.args = args
        self.conn_array = conn_array
        self.side = side
        self.device = device
        self.thread_id = thread_id

    def run(self):
        threadLock = threading.Lock()

        threadLock.acquire()
        # 释放锁，开启下一个线程
        threadLock.release()

        return

class Main_process():
    """ 理论上每个进程下可以多线程(多环境)运行，结构中构造相应接口，但是目前各进程实际只跑一个环境"""
    def __init__(self, args):
        self.args = args
        self.conn_list = []
        self.side = 0


    def run(self):

        start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        """ Scenario:
        2  2v2，
        6  6v6  """
        self.args.Scenario(2)

        self.side = 0

        """输入设定的训练进程数量、收集进程数量、评估进程数量，以及平台提供的GPU数量"""
        self.args.train_num, self.args.collect_num, self.args.eva_num, self.args.device_num = 1, 4, 0, 2

        """建立pipe通信通道array,共有5种类型：
        1：collect_red——train_red     buffer，     train_red——collect_red     net
        2: collect_blue——train_blue   buffer，     train_blue——collect_blue   net
        3：train_red——train_blue      net   ,      train_blue——train_red      net 
        4: eva_red——train_red         胜率  ，     train_red——eva_red         net
        5：eva_blue——train_blue       胜率  ，     train_blue——eva_blue       net
        
        形状为：(self.train_num + self.eva_num) * (self.collect_num) * (conn1, conn2)"""
        self.conn_array = [[Pipe() for i in range(self.args.collect_num)] for j in range(self.args.train_num + self.args.eva_num)]

        """根据各类进程数量与GPU资源数量，将各个进程按序分配到各个GPU上"""
        self.args.set_process(self.args.train_num, self.args.collect_num, self.args.eva_num, self.args.device_num)

        """建立进程"""
        train_list = [Process(target=self.train, args=(self.args, self.conn_array, self.side, self.args.device_distribute[t_id], t_id, start_time)) for t_id in range(self.args.train_num)]
        collect_list = [Process(target=self.collect, args=(self.args, self.conn_array, self.side, self.args.device_distribute[c_id + self.args.train_num], c_id, start_time)) for c_id in range(self.args.collect_num)]
        eva_list = [Process(target=self.eva, args=(self.args, self.conn_array, self.side, self.args.device_distribute[e_id + self.args.train_num + self.args.collect_num], e_id, start_time)) for e_id in range(self.args.eva_num)]

        process = []
        process.extend(train_list)
        process.extend(collect_list)
        process.extend(eva_list)
        [p.start() for p in process]
        [p.join() for p in process]

    @staticmethod
    def train(args, conn_array, side, device, id, start_time):

        print("train_id: ", id, "  train_start_time:  ", start_time)
        writer = SummaryWriter(args.tensorboard_savepath+start_time)
        scores = deque(maxlen=100)
        net = Network(args)

        """训练类型：0.单对手强化测试， 1.多对手强化集群训练， 2.基于人工策略基础上的进化训练， 3.纯self-play进化训练"""
        if args.train_type == 2:
            net.load(args.initial_agent_savepath, side, start_time)

        for c_id in range(args.collect_num):
            conn_array[side][c_id][1].send(net)
            time.sleep(0.1)

        """我方需要训练的智能体"""
        our_agent = Agent(args, side, net, device)
        our_agent.start_time = start_time

        total_train_step, single_step = 0, 0

        for episode in count():
            for c_id in range(args.collect_num):

                time.sleep(0.1)
                """从conn_array中依次读取采集到的数据，进行训练"""
                if conn_array[side][c_id][1].poll() == True:
                    memory = conn_array[side][c_id][1].recv()
                    our_agent.memory.reset()
                    our_agent.learn(memory)
                    single_step += 1

                    """本局结束"""
                    if memory.result[1] == 1:
                        """统计胜率"""
                        scores.append(memory.result[0])
                        total_train_step += 1
                        print("train_id: ", id, "  collect_id: ", c_id, "  total_train_step: ", total_train_step)
                        writer.add_scalar('win_rate', Average(scores), global_step=total_train_step)

            """完成过至少十次训练"""
            if single_step >= 10:
                print("train_id: ", id, "  train_num: ", episode, '  win_rate: ', Average(scores))
                single_step = 0
                """向各个collect进程回传刚训练完的网络"""
                net.actor_eval = deepcopy(our_agent.actor_eval).to(torch.device("cpu"))
                net.critic_eval = deepcopy(our_agent.critic_eval).to(torch.device("cpu"))

                for c_id in range(args.collect_num):
                    time.sleep(0.1)
                    conn_array[side][c_id][1].send(net)

                """定期保存网络"""
                if (episode) % args.save_main == 0:
                    net.save(args.main_agent_savepath, args.the_side[side], start_time, episode)

    @staticmethod
    def collect(args, conn_array, side, device, id, start_time):

        print("collect_id: ", id, "  collect_start_time:  ", start_time)

        """调用conn_array中对应id的pipe，0用于collect向train、eva发送，1用于train、eva向collect发送"""
        send_port, recv_port = conn_array[side][id][0], conn_array[side][id][0]

        """从训练进程中获取网络"""
        net = recv_port.recv()

        """环境"""
        run_envs = ENV(args, side, start_time)

        """我方需要训练的智能体"""
        our_agent = Agent(args, side, net, device)
        our_agent.start_time = start_time

        """敌方智能体及其类型：
        获取rule智能体id/获取rl智能体网络， 智能体类型：rule/rl"""
        enemy, enemy_type = run_envs.select_enemy(conn_array, id)

        if enemy_type == "rule":
            """敌方rule智能体从文件夹路径中读取，并初始化"""
            enemy_agent = my_import(f"{args.RuleAgentsPackage}{enemy}")()
        else:
            """敌方rl智能体从conn_array中读取网络并建立智能体"""
            enemy_agent = Agent(args, 1-side, enemy, device)

        for eps in count():
            """环境初始化"""
            enemy_agent.setup(args.scen, 1 - run_envs.side)
            obs, state, enemystate, allstate, done, info, enemy_info = run_envs.reset()

            """运行"""
            for i in count():
                our_action, action_pro = our_agent.act(state, info)
                enemy_action = enemy_agent.step(obs[1-side]) if enemy_type == "rule" else enemy_agent.act(enemystate, enemy_info)[0]
                next_obs, next_state, next_enemystate, next_allstate, reward, done, info, enemy_info = run_envs.step(our_action, enemy_action, enemy_type)

                """记录数据并传至train进程中"""
                our_agent.step(i, state, our_action, reward, done, action_pro, allstate, next_allstate)

                obs, state, enemystate, allstate = next_obs, next_state, next_enemystate, next_allstate

                if our_agent.send == 1:
                    """记录结果【结果、对局结束标志位】"""
                    our_agent.memory.result = run_envs.result

                    """发送采集数据与对局结果"""
                    send_port.send(our_agent.memory)
                    our_agent.memory.reset()
                    print("collect_id: ", id, "  collect_send:   ", i+1)
                    if done == True:
                        end_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                        print("collect_id: ", id, "  collect_num: ", eps, "  collect_end_time:  ", end_time)
                        if args.replay == 1:# and (eps + 1) % 100 == 0:
                            run_envs.env.save_replay(str(id) + "___" + str(end_time))

                        """运行完一局，进行一次网络更新"""
                        while recv_port.poll():
                            """重复查询conn_array当前来自train进程的最新网络，不断进行读取操作，直至读取至最新的那个"""
                            net = recv_port.recv()
                            time.sleep(0.1)
                        our_agent.load_net(net)
                        break


    @staticmethod
    def eva(args, conn_list, side, device, id, start_time):
        return


if __name__ == "__main__":
    args = arguments()
    Mainrun = Main_process(args)
    Mainrun.run()
