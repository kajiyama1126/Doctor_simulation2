# -*- coding: utf-8 -*-
import copy

import networkx as nx
import numpy as np


class Communication:  # (頂点数，辺数，辺確率)
    def __init__(self, n, k, p):
        self.n = n
        self.k = k
        self.p = p
        self.count = 0

    def make_connected_WS_graph(self):
        self.G = nx.connected_watts_strogatz_graph(self.n, self.k, self.p)
        #         lam = nx.laplacian_spectrum(G)
        #         print(nx.adjacency_matrix(G))
        #         print (number_of_nodes(G))
        #         (nx.degree(G))
        # print(self.G)
        self.A = np.array(nx.adjacency_matrix(self.G).todense())  # 隣接行列

    #         print(self.A)

    # def make_graph(self,number):
    #     graph = [nx.dense_gnm_random_graph(self.n,self.m) for i in range(number)]


    def divide_graph(self, divide):
        self.divide_number = divide

        m1 = [int(4 * i) for i in range(13)]
        m2 = [int(4 * i + 1) for i in range(13)]
        m3 = [int(4 * i + 2) for i in range(12)]
        m4 = [int(4 * i + 3) for i in range(12)]
        m = [m1, m2, m3, m4]
        # m = self.n/self.divide_number
        # pattern = [[int(self.divide_number * i + j)  for i in range()] for j in range(self.divide_numbervide)]
        self.G_divide = [copy.copy(self.G.subgraph(m[i])) for i in range(divide)]

        self.A_div = [np.array(nx.adjacency_matrix(self.G_divide[i]).todense()) for i in range(divide)]

    def weight_matrix_div(self, number):
        a = np.zeros(self.n)
        for i in range(self.n):
            a[i] = copy.copy(1.0 / (nx.degree(self.G_divide[number])[i] + 1.0))

        self.P = np.zeros((self.n, self.n))  # 確率行列(重み付き)
        for i in range(self.n):
            for j in range(i, self.n):
                if i != j and self.A_div[number][i][j] == 1:
                    a_ij = min(a[i], a[j])
                    self.P[i][j] = copy.copy(a_ij)
                    self.P[j][i] = copy.copy(a_ij)


                    #         print(self.P)
        for i in range(self.n):
            sum = 0.0
            for j in range(self.n):
                sum += self.P[i][j]
            self.P[i][i] = 1.0 - sum

    def weight_martix(self):
        a = np.zeros(self.n)
        for i in range(self.n):
            a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 1.0))

        self.P = np.zeros((self.n, self.n))  # 確率行列(重み付き)
        for i in range(self.n):
            for j in range(i, self.n):
                if i != j and self.A[i][j] == 1:
                    a_ij = min(a[i], a[j])
                    self.P[i][j] = copy.copy(a_ij)
                    self.P[j][i] = copy.copy(a_ij)


                #         print(self.P)
        for i in range(self.n):
            sum = 0.0
            for j in range(self.n):
                sum += self.P[i][j]
            self.P[i][i] = 1.0 - sum
