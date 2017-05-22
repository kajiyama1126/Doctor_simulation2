# -*- coding: utf-8 -*-
import copy

import numpy as np

from agent_src.agent import Agent_i_constrain


# from databox import Trigger_data


class Agent_i_constrain_event(Agent_i_constrain):
    def __init__(self, n, m, weight, system_A, output_b, number, s, R, th,th_pa):
        super(Agent_i_constrain_event, self).__init__(n, m, weight, system_A, output_b, number, s, R)
        self.tildex_ij = np.zeros((self.n, self.m, 1))
        self.tildex_ji = np.zeros((self.n, self.m, 1))
        self.trigger = np.zeros(self.n)
        self.E = th
        self.trigger_check = np.ones(n)
        self.trigger_count = np.zeros(self.n)
        self.trigger_time = [[] for i in range(n)]
        self.trigger_time2 = [[]for i in range(n)]
        self.neighbor_count = np.sum(np.sign(weight)) - 1
        # self.step = s
        # self.neighbor = self.neighbor_check
        self.th_pattern = th_pa
        # self.trigger_data = Trigger_data

        self.neighbor = []
        self.neighbor_check()
        self.graph_trigger_count = 0

    def neighbor_check(self):
        for i in range(self.n):
            if i != self.name:
                if self.a[i] > 0:
                    self.neighbor.append(i)

    def threshold(self, k,j):
        # if self.th_pattern == 0:
        #     if self.a[j] > 0.15:
        #         return self.s(k) * self.E
        #     else:
        #         return 2*self.s(k)*self.E
        if self.th_pattern == 0:
            return 0

        elif self.th_pattern == 1:
            return 1.0/((k+1))

        elif self.th_pattern == 2 :
            return 10.0/((k+1))

        elif self.th_pattern == 3:
            return 100.0/((k+1))

        elif self.th_pattern == 4:
            return 10.0 / ((k + 1)**2)

        elif self.th_pattern == 5:
            return 10.0 / ((k + 1) ** 0.7)

        elif self.th_pattern == 6:
            return 10.0 / ((k + 1) ** 0.5)
            # if k == 1:
            #     self.th1 = 1.0
            # else:
            #     self.th1 = self.th1 * 0.99
            # return self.th1


        # else:
        #     if self.a[j] > 0.15:
        #         return self.E * 0.995**k
        #     else:
        #         return 2*self.E * 0.995**k



    def trigger_judge(self, k):
        for j in range(self.n):
            if self.a[j] > 0:
                if np.linalg.norm(self.x_i - self.tildex_ij[j]) >= self.threshold(k,j):
                    self.trigger_check[j] = 1
                else:
                    self.trigger_check[j] = 0

    def send(self, j):
        if self.graph_trigger_count == self.neighbor_count:
            self.graph_trigger_count = 1
        else:
            self.graph_trigger_count += 1

        if self.trigger_check[j] == 1:
            self.tildex_ij[j] = copy.copy(self.x_i)
            self.trigger_count[j] += 1
            self.trigger_time[j].append(self.graph_trigger_count)
            self.trigger_time2[j].append(-10)
            return self.x_i
        else:
            self.trigger_time[j].append(-10)
            self.trigger_time2[j].append(self.graph_trigger_count)
            return None

    def receive(self, j, x_j):
        if x_j is None:
            return
        else:
            self.tildex_ji[j] = x_j

    def koshin(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.tildex_ji[j] - self.tildex_ij[j])
        z_i = self.x_i + sum - (self.s(k) * self.d_i())
        self.x_i = self.P_X(z_i)
        self.trigger_judge(k + 1)

class Agent_i_constrain_event_TAC_step_fix(Agent_i_constrain_event):
    def threshold(self, k,j):
        # if self.th_pattern == 0:
        #     if self.a[j] > 0.15:
        #         return self.s(k) * self.E
        #     else:
        #         return 2*self.s(k)*self.E
        if self.th_pattern == 0:
            return 0

        elif self.th_pattern == 1:
            return 1.0/((k+1))

        elif self.th_pattern == 2 :
            return 10.0/((k+1))

        elif self.th_pattern == 3:
            return 40.0/((k+1))

        elif self.th_pattern == 4:
            return 10.0 / ((k + 1)**2)

        elif self.th_pattern == 5:
            return 10.0 / ((k + 1) ** 0.75)

        elif self.th_pattern == 6:
            return 0.2
            # return 10.0 / ((k + 1) ** 0.5)

class Agent_i_constrain_event_TAC_thresh_fix(Agent_i_constrain_event):
    def __init__(self, n, m, weight, system_A, output_b, number, s, R, th,th_pa):
        super(Agent_i_constrain_event_TAC_thresh_fix, self).__init__(n, m, weight, system_A, output_b, number, s, R,th,th_pa)
        self.step_pattern = th_pa
    def threshold(self, k,j):
        return 10.0/((k+1))

    def s(self,k):
        if self.th_pattern == 0:
            return 0.50/((k+1))

        elif self.th_pattern == 1:
            return 1.0/((k+1))

        elif self.th_pattern == 2:
            return 2.0/((k+1))

        elif self.th_pattern == 3:
            return 5.0 / ((k + 1))

        elif self.th_pattern == 4:
            return 1.0 / ((k + 1) ** 0.8)

        elif self.th_pattern == 5:
            return 1.0 / ((k + 1) ** 0.6)



class Agent_i_constrain_event_moment(Agent_i_constrain_event):
    def __init__(self, n, m, weight, system_A, output_b, number, s, R, th,th_pa = 0):
        super(Agent_i_constrain_event_moment, self).__init__(n, m, weight, system_A, output_b, number, s, R, th, th_pa)
        self.gamma = 0.999
        self.v_i = np.zeros((self.m,1))


    def hat_s(self, k):  # ステップ幅
        # return 1.0
        return 10.0 / (k + 100) ** 0.51

    def v_update(self, k):
        self.v_i = self.hat_s(k)*self.gamma*self.v_i + self.s(k)*self.d_i()


    def koshin(self, k):
        self.v_update(k)
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.tildex_ji[j] - self.tildex_ij[j])
        z_i = self.x_i + sum
        z_i = z_i - self.v_i
        self.x_i = z_i
        self.x_i = self.P_X(self.x_i)
        self.trigger_judge(k + 1)

class Agent_i_constrain_event_moment_psi(Agent_i_constrain_event_moment):
    def __init__(self, n, m, weight, system_A, output_b, number, s, R, th,th_pa = 0):
        super(Agent_i_constrain_event_moment_psi, self).__init__(n, m, weight, system_A, output_b, number, s, R, th, th_pa)
        self.v_i = self.d_i()
        self.psi_i = np.zeros((self.n,self.m,1))
        self.psi_j = np.zeros((self.n, self.m, 1))



    def koshin(self, k):
        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.tildex_ji[j] - self.tildex_ij[j])
        z_i = self.x_i + sum
        z_i = z_i - self.s(k) * self.v_i
        self.x_i = z_i
        self.x_i = self.P_X(self.x_i)
        self.v_update()
        self.trigger_judge(k + 1)


    def send(self, j):
        if self.graph_trigger_count == self.neighbor_count:
            self.graph_trigger_count = 1
        else:
            self.graph_trigger_count += 1

        if self.trigger_check[j] == 1:
            self.tildex_ij[j] = copy.copy(self.x_i)
            self.trigger_count[j] += 1
            self.trigger_time[j].append(self.graph_trigger_count)
            self.trigger_time2[j].append(-10)

            self.psi_i[j] = self.v_i
            return self.x_i, self.v_i
        else:
            self.trigger_time[j].append(-10)
            self.trigger_time2[j].append(self.graph_trigger_count)
            self.tildex_ij[j] = self.P_X(self.tildex_ij[j] + self.psi_i[j])
            return None

    def receive(self, j, x_j,psi):
        if x_j is None:
            self.tildex_ji = self.P_X(self.tildex_ji + self.psi_j[j])
            return
        else:
            self.tildex_ji[j] = x_j
            self.psi_j[j] = psi


    def v_update(self):
        self.v_i = self.gamma*self.v_i + self.d_i()


    def trigger_judge(self, k):
        for j in range(self.n):
            if self.a[j] > 0:
                if np.linalg.norm(self.x_i - (self.tildex_ij[j] + self.psi_i[j])) >= self.threshold(k,j):
                    self.trigger_check[j] = 1
                else:
                    self.trigger_check[j] = 0