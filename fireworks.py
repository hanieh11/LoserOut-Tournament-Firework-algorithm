import numpy as np
import matplotlib.pyplot as plt
from progress.bar import ChargingBar

setting = np.zeros((5,6))
#first col is current fitness 
#second col is ranking
#3rd is spark number
#forth is amplitude
#5th is delta
#6th is previous fitness


#initializing the firework uniformly
def inital_fw(dimention, search_space):
    x = np.zeros(dimention)
    for i in range(dimention):
      x[i] = np.random.uniform(search_space[0], search_space[1])
    return x


def get_fitness(x, f_name):
    return f_name(x)


#fw is the fire work matrix
def rank_fw(fw, f_name): 
    global setting

    fitness = np.zeros(fw.shape[0]) 
    for firework in range(fw.shape[0]):
        fitness[firework] = get_fitness(fw[firework], f_name)
    setting[:, 0] = fitness
    arg = np.argsort(fitness)
    for i,rank in enumerate(arg):
        setting[rank][1] = i+1
    #now the 2nd col of setting is filled with rank


def get_spark_number(landa, alpha):
    global setting

    setting[:,2] = setting[:,1]**(-alpha)
    setting[:,2] = landa * setting[:,2] / np.sum(setting[:,2])
    setting[:,2] = setting[:,2].astype(int)
    

def get_amplitude(c_a, c_r):
    global setting

    for i in range(setting.shape[0]):
        if setting[i][0] >= setting[i][5]: 
            setting[i][3] = c_r * setting[i][3]
        else: setting[i][3] = c_a * setting[i][3]


def explosion(fw, fw_num):
    global setting

    sparks = np.zeros((int(setting[fw_num][2]), fw.shape[1]))
    for j in range(int(setting[fw_num][2])):
        sparks[j] = fw[fw_num] + (setting[fw_num][3] * np.random.uniform(-1,1, size=fw.shape[1]))
    return sparks


def mutation(firework, sparks, mu_rate, f_name):
    fitness = np.zeros(sparks.shape[0])
    for j in range(sparks.shape[0]):
        fitness[j] = get_fitness(sparks[j],f_name)

    fitness = np.argsort(fitness) #sort spark fiteness in ascending order
    selected_pop = np.ceil(mu_rate*sparks.shape[0]) # choose the number of top or bottom sparks
    selected_pop = int(selected_pop)

    top = np.sum(sparks[fitness[:selected_pop],:], axis=0) #sum of best
    bottom = np.sum(sparks[fitness[-selected_pop:],:], axis=0) #sum of worst

    mutated = firework + (1/selected_pop)*(top - bottom) #calculate the guiding spark
    return mutated

def fw_selection(firework, sparks, mutated, f_name):

    fitness = np.zeros(sparks.shape[0]+2)
    for i in range(sparks.shape[0]):
        fitness[i] = get_fitness(sparks[i],f_name)
    fitness[-2] = get_fitness(firework, f_name) #firework itself
    fitness[-1] = get_fitness(mutated, f_name)

    arg = np.argsort(fitness)
    if arg[0] in range(sparks.shape[0]): best = sparks[arg[0]]
    elif arg[0] == sparks.shape[0]: best = firework
    elif arg[0] == sparks.shape[0]+1 : best = mutated
    best_fitness = fitness[arg[0]]
    return best, best_fitness


def tournament(fw, max_gen, current_gen, search_space, f_name, init_delta, best):
    global setting

    setting[:,4] = np.where(setting[:,5]>setting[:,0], setting[:,5]-setting[:,0], setting[:,4])

    for i in range(setting.shape[0]):

        if (setting[i][4]*(max_gen-current_gen)) < (setting[i][0]-best):
            # print('one loser was found :', i)
            setting[i][5] = setting[i][0]
            fw[i] = inital_fw(fw.shape[1], search_space)
            setting[i][4] = init_delta
    return fw

def run_LoFW(f_name, dimentions, search_space):
    global setting

    population = 5
    n_generation = 500
    generation = 1
    landa = 300
    alpha = 0.6
    inital_delta = 1
    mu_rate = 0.2
    best = np.zeros((n_generation,population))
    
    # bar = ChargingBar('Proessing', max = n_generation)

    #we run the first generation to set the inital parametrs for the rest of generations
    fw = np.zeros((population, dimentions))
    for i in range(population):
        fw[i] = inital_fw(dimentions, search_space)

    rank_fw(fw, f_name)
    get_spark_number(landa, alpha)
    setting[:,3] = dimentions
    setting[:,4] = inital_delta
    setting[:,5] = setting[:,0] #initialing the 1st generation fitness as the g-1 fitness for 2nd generation for further use

# create sparks and guiding spark to firework for the first gen and pick the fittest

    for i in range(fw.shape[0]):
        sparks = explosion(fw, i)
        mutated = mutation(fw[i], sparks, mu_rate, f_name)
        fw[i], best[0][i] = fw_selection(fw[i], sparks, mutated, f_name)

    while generation < n_generation:
        # bar.next()
        fw , generation_best, best[generation] = run_fireworks(f_name, dimentions, search_space, fw, landa, alpha, mu_rate, n_generation, generation, inital_delta, population)
        generation += 1 ######caution starts from 1 till 499
        
    # bar.next()
    # bar.finish()
    return generation_best, best



def run_fireworks(f_name, dimentions, search_space, fw, landa, alpha, mu_rate, max_generation, generation, delta, population):
    global setting

    c_a = 1.2
    c_r = 0.9
    global_best = 1400000
    best = np.zeros(population)
    best = best + 1400000

    rank_fw(fw, f_name)
    get_spark_number(landa, alpha)
    get_amplitude(c_a, c_r)
    setting[:,5] = setting[:,0] #initialing the 1st generation fitness as the g-1 fitness for 2nd generation for further use
    # print('---------------------------------',generation,'---------------------------')

    for i in range(fw.shape[0]):
        sparks = explosion(fw, i)
        # plt.plot(sparks[:, 0], sparks[:, 1], 'r.')
        mutated = mutation(fw[i], sparks, mu_rate, f_name)
        fw[i], best[i] = fw_selection(fw[i], sparks, mutated, f_name)

        if best[i] < global_best:
            global_best = best[i]

    rank_fw(fw, f_name)
    fw = tournament(fw, max_generation, generation, search_space, f_name, delta, global_best)
    return fw , global_best, best


