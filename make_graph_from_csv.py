import configparser
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file = '20170518_123358'
os.chdir('data')
os.chdir(file)

config = configparser.ConfigParser()
config.read(file + '.txt')

test_patter = int(config['simulation']['normal'])
test_event = int(config['simulation']['event'])
test_D_NG = int(config['simulation']['D_NG'])
test_nedic = int(config['simulation']['nedic'])
test = (test_patter, test_event, test_D_NG, test_nedic)

f_value = pd.read_csv(file + 'f_value' + '.csv')

th_pa = [0, 0, 0, 1, 1, 1, 2, 2, 2,3,3,3]
stepsize = []
for i in range(3):
    stepsize.append(float(config['stepsize']['stepsize' + str(int(i + 1))]))

ave_f = [[[] for i in range(j)] for j in test]

tmp2 = 0
for i2 in range(len(test)):
    for i1 in range(test[i2]):
        tmp2 += 1
        ave_f[i2][i1] = f_value.iloc[:, tmp2]

color = ['b', 'r', 'y']
color =['black','r','b','g','m','c','y']
line = ['-', '--', '-.', ':']
step_index = ['$s(k) = 0.5/(k+1)$', '$s(k) = 1.0/(k+1)$', '$s(k) = 2.0/(k+1)$']
step_index2 = [r'$s_{1}(k)$', r'$s_{2}(k)$', r'$s_{3}(k)$']
# trigger_index = ['$E_i(k) = 0, for all i', '$E_i(k) = 0, for all i', '$E_i(k) = 0, for all i',
#                  '$E_i(k) = 10/(k+1), for all i', '$E_i(k) = 10/(k+1), for all i', '$E_i(k) = 10/(k+1), for all i',
#                  '$E_i(k) = 10/(k+1)^2, for all i', '$E_i(k) = 10/(k+1)^2, for all i',
#                  '$E_i(k) = 10/(k+1)^2, for all i',
#                  '$E_i(k) = 10*0.99^k, for all i', '$E_i(k) = 10*0.99^k, for all i',
#                  '$E_i(k) = 10*0.99^k, for all i', ]
graph_name_index = ['$E(k) = 0$', '$E(k)=1.0/(k+1)$', '$E(k)=10/(k+1)$', '$E(k)=40/(k+1)$',
                    '$E(k)=10/(k+1)^2$', '$E(k)=10/(k+1)^{0.75}$', '$E(k)=0.2$']
trigger_index2 = [['1'], ['2', '3', '4']]
for i2 in range(len(test)):
    for i1 in range(test[i2]):
        if i2 == 0:
            trigger_name = 'time'
            tmp_line = line[i2]
        elif i2 == 1:
            trigger_name = 'event'
            tmp_line = line[int(th_pa[i1] + 1)]
        plt.plot(ave_f[i2][i1],color = color[i1],linewidth = 1,
                 label=graph_name_index[i1])
# plt.legend()
plt.xlabel('iteration $k$', fontsize=14)
plt.ylabel('$Σ_{i=1}^{50} (f(x_i(k))-f^*)/ Σ_{i=1}^{50}(f(x_i(0))-f^*)$', fontsize=14)
plt.tick_params(labelsize=14)
plt.yscale("log")
plt.ylim([0.9*10**(-4),1])
# plt.ylim([10**(-4),10**(-3)])
plt.legend(fontsize = 16)
sns.set_style("dark")
plt.savefig(file + 'f_value' + ".png")
plt.show()
