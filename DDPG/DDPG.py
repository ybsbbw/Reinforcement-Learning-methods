import gym
import torch
import numpy as np
import random
from collections import namedtuple, deque
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
from DDPG_Agent import Agent

import torch.nn as nn
import torch.nn.functional as F
import datetime
import logging
import time


random.seed(0)
env = gym.make('Pendulum-v0')
env.seed(0)

state_typesize = env.observation_space.shape[0]
action_typesize = env.action_space.shape[0]
#action_typesize = env.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


eps_start = 0.95
eps_end = 0.05
eps_iteration = 0.95

BATCH_SIZE = 256
BUFFER_SIZE = int(1e5)

P_LR = 0.0005
P_GAMMA = 0.99
P_TAU = 1e-3


###run 01
# c01 = int(100)
# n_episode = 10001
# test_episode = 50

###run 02
c01 = int(10)
n_episode = 1001
test_episode = 10

###debug
# c01 = int(5)
# n_episode = 51
# test_episode = 2

render = False
DEBUG = True
#print("env.observation_space.shape[0]==",env.observation_space.shape[0])
#print("env.action_space.n==",env.action_space.n)

dir = "C:\\Users\\a2\Desktop\\projects\\Pendulum-v0\\DDPG_ZWS"
taddr = time.strftime("---%Y_%m_%d_%H_%M_%S", time.localtime())

saveaddressA = dir+'\\traindata\\train_model\\Actore-net\\' + str(n_episode) + taddr
saveaddressA_raw = saveaddressA+'\\raw'

saveaddressC = dir+'\\traindata\\train_model\\Critic-net\\' + str(n_episode) + taddr
saveaddressC_raw = saveaddressC+'\\raw'

lossa_saveaddress = dir + '\\traindata\\lossesA_average\\' + str(n_episode) + taddr
lossc_saveaddress = dir + '\\traindata\\lossesC_average\\' + str(n_episode) + taddr

score_saveaddress = dir + '\\traindata\\scores_average\\' + str(n_episode) + taddr

image_saveaddress = dir + '\\image\\' + str(n_episode) + taddr
test_saveaddress = dir + '\\testdata\\' + str(n_episode) + taddr


def Average(array):
    avg = 0.0
    n = len(array)
    for num in array:
        avg += 1.0*num/n
    return avg

def train(episode, trainrender = False, LR = 0.0005, GAMMA = 0.99, TAU = 1e-3):
    trainagent = Agent(state_typesize, action_typesize, device, BATCH_SIZE, BUFFER_SIZE, LR, GAMMA, TAU)

    if os.path.exists(dir+'\\traindata\\load'):
        shutil.rmtree(dir+'\\traindata\\load')
        os.mkdir(dir+'\\traindata\\load')
    else:
        os.mkdir(dir + '\\traindata\\load')

    lossesC = deque(maxlen=c01)
    lossesA = deque(maxlen=c01)

    lossesC_average = []
    lossesA_average = []

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

            lossC, lossA = trainagent.step(state, action, reward, next_state, done)

            if lossC and lossA is not None:
                lossesC.append(lossC.cpu().detach().numpy())
                lossesA.append(lossA.cpu().detach().numpy())
#                print('\rSteps: {}, loss average: {}'.format(steps, loss))
            if trainrender is True:
                env.render()

            state = next_state

            score += reward

            if done:
                # print('\rTrainDone: episode: {}, step:{}, state: {}, action: {}, reward: {}, score: {}'.format(episode, step, state, action, reward, score), end='')
                break
        print('\rTRAIN: episode: {}, totalsteps: {}, score: {}'.format(i, steps, score), end=' ')

        if score >= -200:
            print("=========>  SUCCESS")
        else:
            print("   ")

        scores.append(score)
        score_average = Average(scores)

        if i % c01 == 0 and i >= 0:
            torch.save(trainagent.actor_eval.state_dict(),
                       dir + '\\traindata\\load\\DDPG_actor_eval_train' + '_' + str(i) + '.pth')
            if DEBUG == False:

                if os.path.exists(saveaddressA) is False:
                    os.mkdir(saveaddressA)
                    os.mkdir(saveaddressA_raw)

                if os.path.exists(saveaddressC) is False:
                    os.mkdir(saveaddressC)
                    os.mkdir(saveaddressC_raw)

                torch.save(trainagent.actor_eval.state_dict(), saveaddressA_raw + '\\DDPG_actor_eval_train_' + 'LR_{}, GAMMA_{}, TAU_{}'.format(LR, GAMMA, TAU) + '_ ' + str(i) + '_ ' + str(score_average) + '.pth')

                torch.save(trainagent.critic_eval.state_dict(), saveaddressC_raw + '\\DDPG_critic_eval_train_' + 'LR_{}, GAMMA_{}, TAU_{}'.format(LR, GAMMA, TAU) + '_ ' + str(i) + '_ ' + str(score_average) + '.pth')

        if DEBUG == False:
            if score_average >= old_score_average:
                torch.save(trainagent.actor_eval.state_dict(), saveaddressA+'\\DDPG_actor_eval_best_' + str(n_episode) + '_LR_{}, GAMMA_{}, TAU_{}'.format(LR, GAMMA, TAU) + '_.pth')

                torch.save(trainagent.critic_eval.state_dict(), saveaddressC+'\\DDPG_critic_eval_best_' + str(n_episode) + '_LR_{}, GAMMA_{}, TAU_{}'.format(LR, GAMMA, TAU) + '_.pth')

                old_score_average = score_average

        scores_average.append(score_average)
        lossesC_average.append(Average(lossesC))
        lossesA_average.append(Average(lossesC))

#        score_average = 0
#        loss_average = 0

        plt.ion()

        plt.subplot(141)
        plt.title('Loss C')
        plt.xlabel('episode')
        plt.ylabel('loss C')
        plt.plot(np.arange(len(lossesC_average)), lossesC_average, 'g^')
       # plt.pause(0.1)
       #      data01 = {'losses_average': losses_average}

        plt.subplot(142)
        plt.title('Loss A')
        plt.xlabel('episode')
        plt.ylabel('loss A')
        plt.plot(np.arange(len(lossesA_average)), lossesA_average, 'g^')

        plt.subplot(143)
        plt.title('Scores-Solve')
        plt.xlabel('episode')
        plt.ylabel('score')
        plt.plot(np.arange(len(scores_average)), scores_average, 'r')
        # data02 = {'scores_average': scores_average}
#        plt.pause()
#        plt.show()
        plt.ioff()

    if DEBUG == False:
        if os.path.exists(lossa_saveaddress) is False:
            os.mkdir(lossa_saveaddress)
        np.savez(lossa_saveaddress + '\\DDPG_train_lossesC_average_' + 'LR_{}, GAMMA_{}, TAU_{}_'.format(LR, GAMMA, TAU) + '.npz',
                 data1=np.arange(len(lossesC_average)), data2=lossesC_average)

        if os.path.exists(lossc_saveaddress) is False:
            os.mkdir(lossc_saveaddress)
        np.savez(lossc_saveaddress + '\\DDPG_train_lossesA_average_' + 'LR_{}, GAMMA_{}, TAU_{}_'.format(LR, GAMMA, TAU) + '.npz',
                 data1=np.arange(len(lossesA_average)), data2=lossesA_average)

        if os.path.exists(score_saveaddress) is False:
            os.mkdir(score_saveaddress)
        np.savez(score_saveaddress + '\\DDPG_train_scores_average_' + 'LR_{}, GAMMA_{}, TAU_{}_'.format(LR, GAMMA, TAU) + '.npz',
                 data1=np.arange(len(scores_average)), data2=scores_average)


def test(episode, testrender=False, LR = 0., GAMMA = 0., TAU = 0.):
    testagent = Agent(state_typesize, action_typesize,  device, BATCH_SIZE, BUFFER_SIZE)
    testscores_average = []
    for j in range(int((n_episode / c01)+1)):
        testagent.actor_eval.load_state_dict(torch.load(dir+'\\traindata\\load\\DDPG_actor_eval_train' + '_' + str(j*c01) + '.pth'))

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

            if testscore >= -200:
                print("=========>  SUCCESS")
            else:
                print("   ")

            testscores.append(testscore)

        testscore_average = Average(testscores)

        testscores_average.append(testscore_average)
#        print('\rTEST: testscore: {}, testscore_average: {}'.format(testscore, testscore_average), end=' ')

        plt.ion()

        plt.subplot(144)
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
            os.mkdir(test_saveaddress)
        np.savez(test_saveaddress + '\\DDPG_test_testscores_average_' + str(n_episode) + '_LR_{}, GAMMA_{}, TAU_{}_'.format(LR, GAMMA, TAU) + '.npz',
                 data1=np.arange(len(testscores_average)), data2=testscores_average)

        if os.path.exists(image_saveaddress) is False:
            os.mkdir(image_saveaddress)
    #    plt.savefig(dir+'\\image\\Normal'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+'.png')
        plt.savefig(image_saveaddress + '\\DDPG_' + str(n_episode) + '_LR_{}, GAMMA_{}, TAU_{}'.format(LR, GAMMA, TAU) + '_.png')
    plt.close()
    env.close()



if __name__ == '__main__':
    ilr = 0.0005
    # igamma = 0.99
    # itau = 1e-3
    if DEBUG == True:
        c01 = int(5)
        n_episode = 51
        igamma = 0.99
        itau = 1e-3
        test_episode = 2
        train(n_episode, render, ilr, igamma, itau)    # 训练
        test(test_episode, render, ilr, igamma, itau)  # 测试
    else:
        for itau in [1e-3, 0.01, 0.1, 0.2, 0.5]:
            for igamma in [0.5, 0.8, 0.9, 0.99, 1, 1.1, 1.5, 2, 5, 10]:
                train(n_episode, render, ilr, igamma, itau)       #训练多少次
                test(test_episode, render, ilr, igamma, itau)     #对每个网络模型测试多少次

    # train(n_episode, render, ilr, igamma, itau)       #训练多少次
    # test(test_episode, render, ilr, igamma, itau)     #对每个网络模型测试多少次