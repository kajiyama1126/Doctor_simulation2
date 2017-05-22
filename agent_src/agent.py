# -*- coding: utf-8 -*-

import numpy as np


class Agent_i(object):
    E = float()

    # (エージェント数，次元数、重み付き行列，A,b，エージェントの名前)
    def __init__(self, n, m, weight, system_A, output_b, number, s):
        self.x_i = np.zeros((m, 1))
        self.x_j = np.zeros((n, m, 1))
        self.a = weight
        self.A = system_A
        self.b = output_b
        self.n = n
        self.m = m

        self.name = number
        self.step = s
        #         self.x_i = np.zeros((m,1))
        # カウント用
        # self.x_i = np.zeros((m, 1))
        self.x_i = 10 * np.random.rand(m, 1)


    def s(self, k):  # ステップ幅
        return self.step / (k + 1)

    def send(self, j):
        return self.x_i

    def receive(self, j, x_j):
        self.x_j[j] = x_j

    def d_i(self):
        A_T = self.A.transpose()
        return 2.0 * np.dot(A_T, (np.dot(self.A, self.x_i) - self.b))

    def koshin(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.x_j[j] - self.x_i)

        z_i = self.x_i + sum - (self.s(k) * self.d_i())
        self.x_i = z_i


class Agent_i_constrain(Agent_i):
    def __init__(self, n, m, weight, system_A, output_b, number, s, R):
        super(Agent_i_constrain, self).__init__(n, m, weight, system_A, output_b, number, s)
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

    def koshin(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.x_j[j] - self.x_i)
        z_i = self.x_i + sum
        self.x_i = z_i
        self.x_i = self.P_X((self.x_i - self.s(k) * self.d_i()))
