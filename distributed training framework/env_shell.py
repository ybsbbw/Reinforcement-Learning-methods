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
import gym
import copy
import random
import numpy as np

from Wargame_land2020.Engine.my_custom_engine.my_env.train_env import TrainEnv
from Wargame_land2020.GeneralEnv.action_translator import ActionTranslator, ActionTranslator6
# from Wargame_land2020.GeneralEnv.feature_extractor import FeatureExtractor, FeatureExtractor6
from Wargame_land2020.GeneralEnv.env_state_shell import FeatureExtractor
# from Wargame_land2020.GeneralEnv.reward_calculator import RewardCalculator, RewardCalculator6
from Wargame_land2020.GeneralEnv.env_reward_shell import RewardCalculator

RED = 0
BLUE = 1

class ActionSortMethod:
    Red_First = 0
    Blue_First = 1
    Random_First = 2

class ENV:
    def __init__(self, args, side, start_time, action_sort_method=ActionSortMethod.Random_First):

        self.args = args
        env_config = {
            "Scenario": self.args.scen,
            "SaveReplay": True if self.args.replay == 1 else False,
            "ReplayPath": self.args.replay_savepath + start_time
        }

        self.env = TrainEnv(env_config)

        self.side = side

        """动作转换"""
        self.action_translator = ActionTranslator(scenario=self.args.scen, color=side, env=self.env, config=None)

        """状态转换"""
        self.feature_extractor = FeatureExtractor(self.args, self.env, side)

        """奖惩计算"""
        self.reward_calculator = RewardCalculator(self.args, self.env, side)

        """ 1:win,  0.5:draw,  0:lose
            0：未结束， 1：已结束    """
        self.result = [0] * 2

        self.action_sort_method = action_sort_method

        self.obs = dict()

    def reset(self):

        self.result = [0] * 2

        self.feature_extractor.reset()
        self.action_translator.reset()
        self.reward_calculator.reset()

        obs, done = self.env.reset()
        state, enemystate, allstate = self.feature_extractor.acquire_state(obs)

        info = {"valid_actions": self.action_translator.translate_valid_actions(obs, self.side)}
        enemy_info = {"valid_actions": self.action_translator.translate_valid_actions(obs, 1 - self.side)}
        self.obs = obs

        return obs, state, enemystate, allstate, done, info["valid_actions"], enemy_info["valid_actions"]


    def step(self, our_action, enemy_action, enemy_type):
        if enemy_type == "rule":
            our_action = self.action_translator.translate_action(self.obs, self.side, our_action)
        else:
            our_action = self.action_translator.translate_action(self.obs, self.side, our_action)
            enemy_action = self.action_translator.translate_action(self.obs, 1-self.side, enemy_action)

        action = self.output_action(our_action, enemy_action)
        next_obs, done = self.env.step(action)
        next_state, next_enemystate, next_allstate = self.feature_extractor.acquire_state(next_obs)

        reward = self.reward_calculator.get_rewards(next_obs)
        info = {"valid_actions": self.action_translator.translate_valid_actions(next_obs, self.side)}
        enemy_info = {"valid_actions": self.action_translator.translate_valid_actions(next_obs, 1 - self.side)}
        if done:
            self.result[1] = 1
            if self.side == 0 and next_obs[self.side]['scores']['red_win'] > 0:
                self.result[0] = 1
            elif self.side == 1 and next_obs[self.side]['scores']['blue_win'] > 0:
                self.result[0] = 1
            elif next_obs[self.side]['scores']['red_win'] == next_obs[self.side]['scores']['blue_win']:
                self.result[0] = 0.5
            else:
                self.result[0] = 0

        self.obs = next_obs

        return next_obs, next_state, next_enemystate, next_allstate, reward, done, info["valid_actions"], enemy_info["valid_actions"]


    def update_enemy(self):
        return

    def render(self):
        return

    def output_action(self, red_action, blue_action):
        """
        :param red_action: list格式
        :param blue_action: list格式
        :return:
        """
        if self.action_sort_method == ActionSortMethod.Red_First:
            result_action = red_action + blue_action
        elif self.action_sort_method == ActionSortMethod.Blue_First:
            result_action = blue_action + red_action
        else:
            if random.random() <= 0.5:
                result_action = red_action + blue_action
            else:
                result_action = blue_action + red_action

        return result_action

    def select_enemy(self, conn_array, id):
        if self.args.train_type == 0: # 2V2 或者 6V6
            enemy_agent, enemy_type = "rule_009", "rule"  # "rule_072"
        elif self.args.train_type == 1 and self.args.scen == 2010131194: # 2V2
            enemy_agent, enemy_type = self.args.agent_list_2v2[id % len(self.args.agent_list_2v2)], "rule"
        elif self.args.train_type == 1 and self.args.scen == 2010431153: # 6V6
            enemy_agent, enemy_type = self.args.agent_list_6v6[id % len(self.args.agent_list_6v6)], "rule"
        elif self.args.train_type == 2 or self.args.train_type == 3:
            enemy_agent, enemy_type = copy.deepcopy(conn_array[1 - self.side][id][1]), "rl"
        else:
            raise ValueError("Wrong train type!!!")

        return  enemy_agent, enemy_type