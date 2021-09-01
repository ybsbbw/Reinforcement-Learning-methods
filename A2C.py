import gym
import torch
import numpy as np
import random
from collections import namedtuple, deque
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
from A2C_Agent import Agent

import torch.nn as nn
import torch.nn.functional as F
import datetime
import logging
import time

random.seed(0)
# env = gym.make('MountainCar-v0') #无法有效收敛
env = gym.make('CartPole-v0') #能够有效收敛

env.seed(0)

#observation_space连续空间：
state_typesize = env.observation_space.shape[0]

#action_space离散空间：
action_typesize = env.action_space.n
#action_typesize = env.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


eps_start = 0.95
eps_end = 0.05
eps_iteration = 0.95

BATCH_SIZE = 256
BUFFER_SIZE = int(1e5)#1e5
LAMDA = 20

P_LR = 0.0005
P_GAMMA = 0.99
P_TAU = 1e-3

#c01: 间隔保存训练网络参数
###run 01
c01 = int(100)
n_episode = 10001
test_episode = 50

###run 02
# c01 = int(10)
# n_episode = 1001
# test_episode = 10

###debug
# c01 = int(5)
# n_episode = 51
# test_episode = 2

render = False
DEBUG = True
#print("env.observation_space.shape[0]==",env.observation_space.shape[0])
#print("env.action_space.n==",env.action_space.n)

# dir = "D:\\researches\\projects and research\\Atari reinforcement learning\\MoutainCar-v0\\A2C_ZWS"
taddr = time.strftime("---%Y_%m_%d_%H_%M_%S", time.localtime())

saveaddress = '../A2C_ZWS/traindata/train_model/Advantage_Actor_Critic/' + str(n_episode) + taddr
saveaddress_raw = saveaddress+'/raw'

loss_saveaddress = '../A2C_ZWS/traindata/losses_average/' + str(n_episode) + taddr

score_saveaddress = '../A2C_ZWS/traindata/scores_average/' + str(n_episode) + taddr

image_saveaddress = '../A2C_ZWS/image/' + str(n_episode) + taddr
test_saveaddress = '../A2C_ZWS/testdata/' + str(n_episode) + taddr


def Average(array):
    avg = 0.0
    n = len(array)
    for num in array:
        avg += 1.0*num/n
    return avg

def train(episode, trainrender = False, LR = 0.0005, GAMMA = 0.99, TAU = 1e-3):
    trainagent = Agent(state_typesize, action_typesize, device, BATCH_SIZE, BUFFER_SIZE, LAMDA, LR, GAMMA, TAU)

    if os.path.exists('../traindata/load'):
        shutil.rmtree('../traindata/load')
        os.makedirs('../traindata/load')
    else:
        os.makedirs('../traindata/load')

    losses = deque(maxlen=c01)

    losses_average = []

    scores = deque(maxlen = c01)
    scores_average = []
    plt.figure(figsize=(16, 8))
    steps = 0
    cycle = 0
    score_average = 0
    old_score_average = -1500

    for i in range(episode):

        score = 0
        state = env.reset()
        step = 0
        eps = eps_start
        while True:
            steps += 1
            step += 1
            if eps >= eps_end:
                eps *= eps_iteration
            action = trainagent.act(state)  #单个的数

            next_state, reward, done, _ = env.step(action)

            the_loss = trainagent.step(state, action, reward, next_state, done)

            if the_loss is not None:
                # losses.append(the_loss.cpu().detach().numpy())
                losses.append(the_loss)
#                print('\rSteps: {}, loss average: {}'.format(steps, loss))
            if trainrender is True:
                env.render()

            state = next_state

            score += reward

            if done:
                # print('\rTrainDone: episode: {}, step:{}, state: {}, action: {}, reward: {}, score: {}'.format(episode, step, state, action, reward, score), end='')
                break
        print('\rTRAIN: episode: {}, totalsteps: {}, score: {}'.format(i, steps, score), end=' ')

        # if score != -200:
        if score >= 195:
            print("=========>  SUCCESS")
        else:
            print("   ")

        scores.append(score)
        score_average = Average(scores)
        if i % c01 == 0 and i >= 0:
            torch.save(trainagent.A2C_eval.state_dict(),
                       '../traindata/load/A2C_eval_train' + '_' + str(i) + '.pth')
            if DEBUG == False:

                if os.path.exists(saveaddress) is False:
                    os.mkdir(saveaddress)
                    os.mkdir(saveaddress_raw)

                torch.save(trainagent.A2C_eval.state_dict(), saveaddress_raw + '/A2C_eval_train_' + 'LR_{}, GAMMA_{}, TAU_{}'.format(LR, GAMMA, TAU) + '_ ' + str(i) + '_ ' + str(score_average) + '.pth')

        if DEBUG == False:
            if score_average >= old_score_average:
                torch.save(trainagent.A2C_eval.state_dict(), saveaddress+'/A2C_eval_best_' + str(n_episode) + '_LR_{}, GAMMA_{}, TAU_{}'.format(LR, GAMMA, TAU) + '_.pth')

                old_score_average = score_average

        scores_average.append(score_average)
        losses_average.append(Average(losses))


#        score_average = 0
#        loss_average = 0

#         plt.ion()
#
#         plt.subplot(131)
#         plt.title('Loss')
#         plt.xlabel('episode')
#         plt.ylabel('loss')
#         plt.plot(np.arange(len(losses_average)), losses_average, 'g^')
#        # plt.pause(0.1)
#        # data01 = {'losses_average': losses_average}
#
#         plt.subplot(132)
#         plt.title('Scores-Solve')
#         plt.xlabel('episode')
#         plt.ylabel('score')
#         plt.plot(np.arange(len(scores_average)), scores_average, 'r')
#         # data02 = {'scores_average': scores_average}
# #        plt.pause()
# #        plt.show()
#         plt.ioff()

    if DEBUG == False:
        if os.path.exists(loss_saveaddress) is False:
            os.makedirs(loss_saveaddress)
        np.savez(loss_saveaddress + '/A2C_train_losses_average_' + 'LR_{}, GAMMA_{}, TAU_{}_'.format(LR, GAMMA, TAU) + '.npz',
                 data1=np.arange(len(losses_average)), data2=losses_average)

        if os.path.exists(score_saveaddress) is False:
            os.makedirs(score_saveaddress)
        np.savez(score_saveaddress + '/A2C_train_scores_average_' + 'LR_{}, GAMMA_{}, TAU_{}_'.format(LR, GAMMA, TAU) + '.npz',
                 data1=np.arange(len(scores_average)), data2=scores_average)


def test(episode, testrender=False, LR = 0., GAMMA = 0., TAU = 0.):
    testagent = Agent(state_typesize, action_typesize,  device, BATCH_SIZE, BUFFER_SIZE, LAMDA)
    testscores_average = []
    for j in range(int((n_episode / c01)+1)):
        testagent.A2C_eval.load_state_dict(torch.load('../traindata/load/A2C_eval_train' + '_' + str(j*c01) + '.pth'))

        testscores = []
    #    plt.figure()
        teststeps = 0
    #    loss_average = 0

        for i in range(episode):

            testscore = 0
            teststate = env.reset()

            while True:
                teststeps += 1

                testaction = testagent.act(teststate)  # 单个的数

                testnext_state, testreward, testdone, _ = env.step(testaction)

                if testrender is True:
                    env.render()

                teststate = testnext_state

                testscore += testreward

                if testdone:
                    break
            print('\rTEST: policy: {}, episode: {}, steps: {}, score: {}'.format(j*c01, i, teststeps, testscore), end=' ')

            # if testscore != -200:
            if testscore >= 195:
                print("=========>  SUCCESS")
            else:
                print("   ")

            testscores.append(testscore)

        testscore_average = Average(testscores)

        testscores_average.append(testscore_average)
#        print('\rTEST: testscore: {}, testscore_average: {}'.format(testscore, testscore_average), end=' ')

        plt.ion()

        plt.subplot(133)
        plt.title('Scores-test')
        plt.xlabel('episode')
        plt.ylabel('score')
        plt.plot(np.arange(len(testscores_average)), testscores_average, 'r')
    #    plt.pause()
    #    data03 = {'testscores_average': testscores_average}

#        plt.show()
        plt.ioff()
    if DEBUG == False:
        if os.path.exists(test_saveaddress) is False:
            os.makedirs(test_saveaddress)
        np.savez(test_saveaddress + '/A2C_test_testscores_average_' + str(n_episode) + '_LR_{}, GAMMA_{}, TAU_{}_'.format(LR, GAMMA, TAU) + '.npz',
                 data1=np.arange(len(testscores_average)), data2=testscores_average)

        if os.path.exists(image_saveaddress) is False:
            os.makedirs(image_saveaddress)
    #    plt.savefig(dir+'\\image\\Normal'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+'.png')
        plt.savefig(image_saveaddress + '/A2C_' + str(n_episode) + '_LR_{}, GAMMA_{}, TAU_{}'.format(LR, GAMMA, TAU) + '_.png')
    plt.close()
    env.close()



if __name__ == '__main__':
    ilr = 0.0005#0.0005
    # igamma = 0.99
    # itau = 1e-3
    if DEBUG == True:
        c01 = int(5)
        n_episode = 100001
        igamma = 0.99#0.99
        itau = 1e-3
        test_episode = 20
        train(n_episode, render, ilr, igamma, itau)    # 训练
        test(test_episode, render, ilr, igamma, itau)  # 测试
    else:
        itau = 1e-3
        igamma = 0.99

        train(n_episode, render, ilr, igamma, itau)
        test(test_episode, render, ilr, igamma, itau)
        for itau in [1e-3, 0.01, 0.1, 0.2, 0.5]:
            for igamma in [0.5, 0.8, 0.9, 0.99, 1, 1.1, 1.5, 2, 5, 10]:
                train(n_episode, render, ilr, igamma, itau)       #训练多少次
                test(test_episode, render, ilr, igamma, itau)     #对每个网络模型测试多少次

    # train(n_episode, render, ilr, igamma, itau)       #训练多少次
    # test(test_episode, render, ilr, igamma, itau)     #对每个网络模型测试多少次