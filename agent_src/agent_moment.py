# -*- coding: utf-8 -*-

import numpy as np

from agent_src.agent import Agent_i


class Agent_i_momentum(Agent_i):
    def __init__(self, n, m, weight, system_A, output_b, number, s):
        super(Agent_i_momentum, self).__init__(n, m, weight, system_A, output_b, number, s)
        self.gamma = 0.9
        self.v_i = self.d_i()
        self.v_j = np.zeros((n, m, 1))
        self.bf_val = (np.linalg.norm(np.dot(self.A, self.x_i) - self.b)) ** 2
        self.val = 0

    def value(self):
        self.val = (np.linalg.norm(np.dot(self.A, self.x_i) - self.b)) ** 2

    def send(self, j):
        return self.x_i, self.v_i

    def receive(self, j, x_j, v_j):
        self.x_j[j] = x_j
        self.v_j[j] = v_j

    def v_update(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.v_j[j] - self.v_i)

        self.v_i = self.gamma * (self.v_i + sum)
        x = (self.s(k) * self.d_i())
        self.v_i = self.v_i + x

    def d_i(self):
        A_T = self.A.transpose()
        return 2.0 * np.dot(A_T, (np.dot(self.A, self.x_i) - self.b))

    def koshin(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.x_j[j] - self.x_i)
        z_i = self.x_i + sum
        z_i = z_i - self.v_i
        self.x_i = z_i
        self.v_update(k)
        # self.value()
        # if self.val>self.bf_val :
        #     self.v_i = np.zeros((self.m,1))
        # self.bf_val = (np.linalg.norm(np.dot(self.A, self.x_i) - self.b)) ** 2


class Agent_i_momentum_kai(Agent_i_momentum):
    def __init__(self, n, m, weight, system_A, output_b, number, s):
        self.first = True
        super(Agent_i_momentum_kai, self).__init__(n, m, weight, system_A, output_b, number, s)

    def d_i(self):
        if self.first:
            self.v_i = 0
            self.first = False

        A_T = self.A.transpose()
        return 2.0 * np.dot(A_T, (np.dot(self.A, (self.x_i - self.v_i)) - self.b))

    def s(self, k):  # ステップ幅
        return self.step / k


class Agent_i_momentum2(Agent_i):
    def __init__(self, n, m, weight, system_A, output_b, number, s):
        self.first = True
        super(Agent_i_momentum2, self).__init__(n, m, weight, system_A, output_b, number, s)
        self.v_i = np.zeros((m, 1))
        self.v_j = np.zeros((n, m, 1))
        self.gamma = 0.9

    def send(self, j):
        return self.x_i, self.v_i

    def receive(self, j, x_j, v_j):
        self.x_j[j] = x_j
        self.v_j[j] = v_j

    def v_update(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.v_j[j] - self.v_i)

        self.v_i = self.gamma * (self.v_i + sum)
        x = (self.s(k) * self.d_i())
        self.v_i = self.v_i + x

    def koshin(self, k):
        self.v_update(k)
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.x_j[j] - self.x_i)
        z_i = self.x_i + sum
        z_i = z_i - self.v_i
        self.x_i = z_i


class Agent_i_momentum3(Agent_i_momentum2):#simulation使用
    def __init__(self, n, m, weight, system_A, output_b, number, s, R):
        super(Agent_i_momentum3, self).__init__(n, m, weight, system_A, output_b, number, s)
        self.gamma = 0.99
        # 凸制約用
        self.c = np.array([[0], [0], [0], [0], [0]])
        self.R = R
        self.x_i = self.P_X(self.x_i)

    def P_X(self, x):
        if np.linalg.norm(x - self.c) <= self.R:
            #             print(np.linalg.norm(x-self.c))
            return x
        else:
            # print('P')
            return self.R * (x - self.c) / np.linalg.norm(x - self.c)

    def hat_s(self, k):  # ステップ幅
        return 10.0 / (k + 100) ** 0.51

    def v_update(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.v_j[j] - self.v_i)

        self.v_i = self.gamma * self.hat_s(k) * (self.v_i + sum)
        x = (self.s(k) * self.d_i())
        self.v_i = self.v_i + x

    def koshin(self, k):
        self.v_update(k)
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.x_j[j] - self.x_i)
        z_i = self.x_i + sum
        z_i = z_i - self.v_i
        self.x_i = z_i
        self.x_i = self.P_X(self.x_i)


