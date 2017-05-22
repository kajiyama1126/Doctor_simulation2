from main_tac_class_step_fix import Program
import numpy as np


sum_calc_count = None
sum_communication_count = None
iteration = 100
for test in range(iteration):
    print(str(test+1) + '回目')
    program = Program()
    if test ==0:
        sum_communication_count = program.presend()
        sum_calc_count = program.presend()

    x,y = program.simulation2()

    for i2 in range(len(program.test)):
        for i1 in range(program.test[i2]):
            sum_calc_count[i2][i1] += x[i2][i1]/iteration
            sum_communication_count[i2][i1] += y[i2][i1]/iteration

# sum_sum_calc_count = np.sum(sum_calc_count,axis=)
print(sum_calc_count)
print(sum_communication_count)

