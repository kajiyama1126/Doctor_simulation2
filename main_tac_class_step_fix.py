# -*- coding: utf-8 -*-
import configparser
import copy
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from agent_src.agent import Agent_i_constrain
from agent_src.agent_event import Agent_i_constrain_event_TAC_step_fix
from other_src.make_communication import Communication
from other_src.solver import Solver


class Program(object):
    def __init__(self):
        os.chdir('/Users/kajiyama/PycharmProjects/TAC0517')
        config = configparser.ConfigParser()
        config.read('inifile.txt')
        today = datetime.datetime.now()
        self.file = "{0:%Y%m%d_%H%M%S}".format(today)
        # print("{0:%Y%m%d_%H%M%S}".format(today))
        try:
            os.chdir('data')
        except:
            os.mkdir('data')
            os.chdir('data')

        os.mkdir(self.file)
        os.chdir(self.file)

        with open(self.file + '.txt', 'w') as configfile:
            config.write(configfile)
            # os.chdir('../')
            # ====================================================初期設定======================================
        self.n = int(config['agent']['number'])  # エージェント数
        self.m = int(config['agent']['dimension'])  # 次元
        self.R = float(config['agent']['constrain'])  # 凸制約

        self.iteration = int(config['simulation']['iteration'])
        # 重み行列兼通信グラフ
        Graph = Communication(self.n, 4, 0.5)
        # Graph = Communication(self.n,2,0.3)
        Graph.make_connected_WS_graph()
        Graph.weight_martix()

        self.weight_matrix = Graph.P
        # nx.draw(Graph.G)
        self.time_graph = int(config['simulation']['time_graph'])

        test_patter = int(config['simulation']['normal'])
        test_event = int(config['simulation']['event'])
        test_D_NG = int(config['simulation']['D_NG'])
        test_nedic = int(config['simulation']['nedic'])

        step_normal = float(config['step']['normal'])
        step_event = float(config['step']['event'])
        step_D_NG = float(config['step']['D_NG'])
        step_nedic = float(config['step']['nedic'])

        self.stop_condition = float(config['stop']['condition'])
        self.stepsize = []
        self.threshold = []
        for i in range(1):
            self.stepsize.append(float(config['stepsize']['stepsize' + str(int(i + 1))]))
        for i in range(7):
            self.threshold.append(float(config['event']['threshold' + str(int(i + 1))]))

        self.th_pa = [0, 1, 2, 3, 4, 5, 6, ]
        # ==================================================================================================
        self.test = (test_patter, test_event, test_D_NG, test_nedic)
        self.stopcheck = [[0 for i in range(j)] for j in self.test]

        self.value_f = [[[[] for i in range(self.n)] for j in range(self.test[k])] for k in range(len(self.test))]
        self.ave_f = [[[] for i in range(j)] for j in self.test]
        self.set_A = []
        self.set_b = []
        print(self.test)
        self.calc_count = [[0 for i in range(j)] for j in self.test]
        self.communication_count = [[0 for i in range(j)] for j in self.test]

        # np.random.seed(0)

    def presimulate(self):
        self.allagent = [[[] for i in range(j)] for j in self.test]
        # A = np.identity(self.m) + 0.1 * np.random.rand(self.m, self.m)
        for i in range(self.n):
            A = np.identity(self.m) + 0.1 * np.random.randn(self.m, self.m)
            b = np.array([[-2.], [-1.], [0.], [1.], [2.]]) + 1.0 * np.random.randn(self.m, 1)
            self.set_A.append(copy.copy(A))
            self.set_b.append(copy.copy(b))

            for i2 in range(len(self.test)):
                for i1 in range(self.test[i2]):
                    if i2 == 0:
                        agent_i = Agent_i_constrain(self.n, self.m, self.weight_matrix[i], A, b, i, self.stepsize[i1],
                                                    self.R)
                    elif i2 == 1:
                        agent_i = Agent_i_constrain_event_TAC_step_fix(self.n, self.m, self.weight_matrix[i], A, b, i,
                                                                       self.stepsize[0], self.R,
                                                                       self.threshold[i1], self.th_pa[i1])
                    elif i2 == 2:
                        agent_i = Agent_i_constrain_event_TAC_step_fix(self.n, self.m, self.weight_matrix[i], A, b, i,
                                                                       self.stepsize[i1 % 3], self.R, self.threshold[0],
                                                                       self.th_pa[i1])
                    elif i2 == 3:
                        agent_i = Agent_i_constrain_event_TAC_step_fix(self.n, self.m, self.weight_matrix[i], A, b, i,
                                                                       self.stepsize[i1 % 3], self.R, self.threshold[0],
                                                                       self.th_pa[i1])
                    self.allagent[i2][i1].append(agent_i)
                    # for i1 in range(test_event):
                    #     agent_i = Agent_i_constrain_event(n, m, weight_matrix[i], A, b, i, stepsize[i1 % 3], R, threshold[0],
                    #                                       th_pa[i1])
                    #     allagent[1][i1].append(agent_i)
                    # for i1 in range(test_D_NG):
                    #     agent_i = Agent_i_constrain_event_moment(n, m, weight_matrix[i], A, b, i, (i1 + 1) * step_D_NG, R,
                    #                                              threshold[0],
                    #                                              th_pa[i1])
                    #     allagent[2][i1].append(agent_i)
                    # for i1 in range(test_nedic):
                    #     agent_i = Agent_i_nedic(n, m, weight_matrix[i], A, b, i, (i1 + 1.0) * step_nedic)
                    #     allagent[3][i1].append(agent_i)
                    # =================================================================================================================

    def centlized_solve(self):
        solver = Solver(self.n, self.m, self.R, self.set_A, self.set_b)
        self.optimal_val = solver.solve()

    def save0(self):
        self.stop = [[0 for i in range(j)] for j in self.test]
        self.stop_base = [[0 for i in range(j)] for j in self.test]
        for i2 in range(len(self.test)):
            for i1 in range(self.test[i2]):
                tmp1 = 0
                for i in range(self.n):
                    tmp = self.optimal(i, i1, i2) - self.optimal_val
                    self.value_f[i2][i1][i].append(tmp)
                    # tmp1 = 1 / self.n * tmp
                    tmp1 += tmp
                self.stop[i2][i1] = tmp1
                self.stop_base[i2][i1] = tmp1
                self.ave_f[i2][i1].append(self.stop[i2][i1] / self.stop_base[i2][i1])

    def optimal(self, i0, i1, i2):
        optimal_val = 0
        x_i = self.allagent[i2][i1][i0].x_i
        for i in range(self.n):
            optimal_val += np.dot((np.dot(self.set_A[i], x_i) - self.set_b[i]).T,
                                  np.dot(self.set_A[i], x_i) - self.set_b[i])
        return optimal_val[0][0]

    def save(self):
        self.stop = [[0 for i in range(j)] for j in self.test]
        for i2 in range(len(self.test)):
            for i1 in range(self.test[i2]):
                if self.stopcheck[i2][i1] == 0:
                    tmp1 = 0
                    for i in range(self.n):
                        tmp = self.optimal(i, i1, i2) - self.optimal_val
                        self.value_f[i2][i1][i].append(tmp)
                        # tmp1 += 1 / self.n * tmp
                        tmp1 += tmp
                    self.stop[i2][i1] = tmp1
                    self.ave_f[i2][i1].append(self.stop[i2][i1] / self.stop_base[i2][i1])
                else:
                    self.ave_f[i2][i1].append(np.NaN)

    def simulate(self):
        self.presimulate()
        self.centlized_solve()
        self.save0()
        self.stop_check(k=0)
        for k in range(self.iteration):
            print(k)
            self.trade()
            self.calc(k)
            self.save()
            self.stop_check(k)

        self.simulation_test()
        self.make_csv()
        self.make_graph()

    def simulation2(self):
        self.presimulate()
        self.centlized_solve()
        self.save0()
        self.stop_check(k=0)
        for k in range(self.iteration):
            # print(k)
            self.trade()
            self.calc(k)
            self.save()
            self.stop_check(k)
        # self.make_csv()
        return self.calc_count, self.communication_count

    def simulation_test(self):
        print(self.calc_count)
        print(self.communication_count)

    def trade(self):
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.weight_matrix[i][j] > 0.0:
                    for i2 in range(len(self.test)):
                        for i1 in range(self.test[i2]):
                            if self.stopcheck[i2][i1] == 1:
                                pass
                            else:
                                state = self.allagent[i2][i1][i].send(j)
                                self.allagent[i2][i1][j].receive(i, state)

    def calc(self, k):
        for i2 in range(len(self.test)):
            for i1 in range(self.test[i2]):
                if self.stopcheck[i2][i1] == 1:
                    pass
                else:
                    for i in range(self.n):
                        self.allagent[i2][i1][i].koshin(k)

    def graph_change(self):
        if self.time_graph == 1:
            Graph = Communication(self.n, 4, 0.3)
            Graph.make_connected_WS_graph()
            Graph.weight_martix()
            weight_matrix = Graph.P
            for i in range(self.n):
                for i2 in range(len(self.test)):
                    for i1 in range(self.test[i2]):
                        self.allagent[i2][i1][i].a = weight_matrix[i]

    def stop_check(self, k):
        for i2 in range(len(self.test)):
            for i1 in range(self.test[i2]):
                if (self.stop[i2][i1] / self.stop_base[i2][i1]) < self.stop_condition and self.stopcheck[i2][i1] == 0:
                    self.stopcheck[i2][i1] = 1
                    trigger_sum = 0
                    if i2 == 0:
                        trigger_sum = k
                    else:
                        for i in range(self.n):
                            trigger_sum += self.allagent[i2][i1][i].trigger_count / self.allagent[i2][i1][
                                i].neighbor_count / self.n
                    trigger_sum = np.sum(trigger_sum)
                    self.calc_count[i2][i1] = k+1
                    self.communication_count[i2][i1] = trigger_sum

    def make_graph(self):
        color = ['b', 'r', 'y']
        line = ['-', '--', '-.', ':']
        step_index = ['$s(k) = 0.5/(k+1)$', '$s(k) = 1.0/(k+1)$', '$s(k) = 2.0/(k+1)$']
        step_index2 = [r'$s_{1}(k)$', r'$s_{2}(k)$', r'$s_{3}(k)$']
        # trigger_index = ['$E_i(k) = 0, for all i', '$E_i(k) = 0, for all i', '$E_i(k) = 0, for all i',
        #                  '$E_i(k) = 10/(k+1), for all i', '$E_i(k) = 10/(k+1), for all i',
        #                  '$E_i(k) = 10/(k+1), for all i',
        #                  '$E_i(k) = 10/(k+1)^2, for all i', '$E_i(k) = 10/(k+1)^2, for all i',
        #                  '$E_i(k) = 10/(k+1)^2, for all i',
        #                  '$E_i(k) = 10*0.99^k, for all i', '$E_i(k) = 10*0.99^k, for all i',
        #                  '$E_i(k) = 10*0.99^k, for all i', ]
        graph_name_index = ['$E(k) = 0$', '$E(k)=1.0/(k+1)$', '$E(k)=10/(k+1)$', '$E(k)=40/(k+1)$',
                            '$E(k)=10/(k+1)^2$', '$E(k)=10/(k+1)^{0.75}$', '$E(k)=0.2$']
        trigger_index2 = [['1'], ['2', '3', '4']]
        for i2 in range(len(self.test)):
            for i1 in range(self.test[i2]):
                if i2 == 0:
                    trigger_name = 'time'
                    # tmp_line = line[i2]
                elif i2 == 1:
                    trigger_name = 'event'
                    # tmp_line = line[int(self.th_pa[i1] + 1)]
                # plt.plot(self.ave_f[i2][i1], label=step_index2[1] + ', Trigger Pattern ' + trigger_index[i1])
                plt.plot(self.ave_f[i2][i1],linewidth = 1, label=graph_name_index[i1])
        # plt.legend()
        plt.xlabel('iteration $k$', fontsize=14)
        plt.ylabel('$Σ_{i=1}^{50} (f(x_i(k))-f^*)/ Σ_{i=1}^{50}(f(x_i(0))-f^*)$', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.yscale("log")
        plt.ylim([10 * (-4), 1])
        plt.legend()
        sns.set_style("dark")
        plt.savefig(self.file + 'f_value' + ".png")
        plt.show()

    # def make_graph2(self):
    #     color = ['b', 'r', 'y']
    #     line = ['-', '--', '-.', ':']
    #     step_index = ['$s(k) = 0.5/(k+1)$','$s(k) = 1.0/(k+1)$','$s(k) = 2.0/(k+1)$']
    #     step_index2 = [r'$s_{1}(k)$',r'$s_{2}(k)$',r'$s_{3}(k)$']
    #     trigger_index = ['$E_i(k) = 0, for all i','$E_i(k) = 0, for all i','$E_i(k) = 0, for all i',
    #                      '$E_i(k) = 10/(k+1), for all i','$E_i(k) = 10/(k+1), for all i','$E_i(k) = 10/(k+1), for all i',
    #                      '$E_i(k) = 10/(k+1)^2, for all i','$E_i(k) = 10/(k+1)^2, for all i','$E_i(k) = 10/(k+1)^2, for all i',
    #                      '$E_i(k) = 10*0.99^k, for all i','$E_i(k) = 10*0.99^k, for all i','$E_i(k) = 10*0.99^k, for all i',]
    #     trigger_index2 = [['1'],['2','3','4']]
    #     for i2 in range(len(self.test)):
    #         for i1 in range(self.test[i2]):
    #             if i2 == 0:
    #                 trigger_name = 'time'
    #                 tmp_line = line[i2]
    #             elif i2 == 1:
    #                 trigger_name = 'event'
    #                 tmp_line = line[int(self.th_pa[i1] + 1)]
    #             plt.plot(self.ave_f[i2][i1], c=color[int(i1 % 3)], linestyle=tmp_line,
    #                      label='Step-Size ' + step_index2[i1%3] + ', Trigger Pattern ' + trigger_index2[i2][math.floor(i1/3)])
    #     # plt.legend()
    #     plt.xlabel('iteration $k$', fontsize=14)
    #     plt.ylabel('$Σ_{i=1}^{50} (f(x_i(k))-f^*)/ Σ_{i=1}^{50}(f(x_i(0))-f^*)$', fontsize=14)
    #     plt.tick_params(labelsize=14)
    #     plt.yscale("log")
    #     plt.legend()
    #     sns.set_style("dark")
    #     plt.savefig(self.file + 'f_value' + ".png")
    #     plt.show()

    def make_csv(self):
        data = pd.DataFrame()
        for i2 in range(len(self.test)):
            for i1 in range(self.test[i2]):
                if i2 == 0:
                    trigger_name = 'time'
                elif i2 == 1:
                    trigger_name = 'event'
                data[str(i1)] = self.ave_f[i2][i1]
                # data[trigger_name + 'step ' + str(self.stepsize[i1 % 3]) + 'trigger ' + str(self.th_pa[i1])] = self.ave_f[i2][i1]
        # print(data)
        data.to_csv(self.file + 'f_value' + '.csv')


    def presend(self):
        return [[0 for i in range(j)] for j in self.test]