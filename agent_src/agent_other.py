# -*- coding: utf-8 -*-
import copy

import numpy as np

from agent_src.agent import Agent_i


class Agent_i_D_NG(Agent_i):
    def __init__(self, n, m, weight, system_A, output_b, number, s):
        super(Agent_i_D_NG, self).__init__(n, m, weight, system_A, output_b, number, s)
        self.y_i = copy.copy(self.x_i)
        self.y_j = np.zeros_like(self.x_j)

    def s(self, k):
        return self.step / (k)

    def beta(self, k):
        return (k - 1) / (k + 2)

    def send(self, j):
        return self.y_i

    def receive(self, j, y_j):
        self.y_j[j] = y_j

    def d_i(self):
        A_T = self.A.transpose()
        return 2.0 * np.dot(A_T, (np.dot(self.A, self.y_i) - self.b))

    def koshin(self, k):
        sum = 0.0
        x_i_bf = self.x_i
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.y_j[j] - self.y_i)

        z_i = self.y_i + sum - (self.s(k) * self.d_i())
        self.x_i = z_i
        self.y_i = self.x_i + self.beta(k) * (self.x_i - x_i_bf)


class Agent_i_nedic(Agent_i):
    def __init__(self, n, m, weight, system_A, output_b, number, s):
        super(Agent_i_nedic, self).__init__(n, m, weight, system_A, output_b, number, s)
        self.y_i = self.d_i()
        self.y_j = np.zeros_like(self.x_j)

    def s(self):
        return self.step

    def send(self, j):
        return (self.x_i, self.y_i)

    def receive(self, j, x_j):
        self.x_j[j] = x_j[0]
        self.y_j[j] = x_j[1]

    def koshin(self, k):
        sum = 0.0
        d = self.d_i()
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.x_j[j] - self.x_i)

        z_i = self.x_i + sum - self.s() * self.y_i
        self.x_i = z_i

        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.y_j[j] - self.y_i)
        z_i = self.y_i + sum + self.d_i() - d
        self.y_i = z_i
