import numpy as np
from functions import *
import fireworks
import matplotlib.pyplot as plt
from progress.bar import ChargingBar

iteration = 50
dimentions = 30
search_space = [-100,100]

# validation, best = fireworks.run_LoFW(sphere, dimentions, search_space)
# for i in range(best.shape[1]):
#     plt.plot(10000*best[:,i], label= 'firework '+ str(i))

# plt.xlim([0,best.shape[0]])
# plt.ylim(0,14000)
# plt.xlabel('Number of Generations')
# plt.ylabel('Evaluation Value')
# plt.legend()
# plt.show()

# bar = ChargingBar('Proessing', max = iteration)

for func in func_list:
    print('  ',func.__name__, '-----')
    gen_fitness = np.zeros(iteration)
    for i in range(iteration):
        validation, best = fireworks.run_LoFW(func, dimentions, search_space)
        gen_fitness[i] = validation * 10000
        bar.next()
    print(' mean fitness : ', np.mean(gen_fitness))
bar.next()
bar.finish()
    