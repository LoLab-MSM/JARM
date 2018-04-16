from jnk3_no_ask1 import model
from simplepso.pso import PSO
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# exp_data = pd.read_csv('../data/exp_data_arrestin_50_correction.csv') # 50% correction
exp_data = pd.read_csv('../data/exp_data_arrestin_normalization.csv') # [0,1] range normalization


# idx_pars_calibrate = [3, 21, 23, 25, 27, 29, 32, 33, 34, 35, 36, 37,  39, 41]
# Kcat is same for phosphorylation with and without arrestin
idx_pars_calibrate = [3, 21, 23, 25, 27, 29, 32, 33, 36, 37,  39, 41]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

# Index of Initial conditions of Arrestin
arrestin_idx = [42]
jnk3_initial_value = model.parameters[45].value


param_values = np.array([p.value for p in model.parameters])
nominal_values = np.array([p.value for p in model.parameters])
xnominal = np.log10(nominal_values[rates_of_interest_mask])

tspan = np.linspace(0, exp_data['Time (secs)'].values[-2], 121)
t_exp_mask = [idx in exp_data['Time (secs)'].values[:-1] for idx in tspan]
solver = ScipyOdeSimulator(model, tspan=tspan)


def display(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    sim = solver.run(param_values=param_values).all

    plt.plot(exp_data['Time (secs)'].values[:-1], sim['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value, color='red', label='pJNK3 by MKK4 sim')
    plt.errorbar(exp_data['Time (secs)'].values[:-1], exp_data['pTyr_arrestin_avg'].values[:-1],
                 exp_data['pTyr_arrestin_std'].values[:-1],
                 linestyle='None', marker='o', capsize=5, color='red', label='pJNK3 by MKK4 exp')
    plt.plot(exp_data['Time (secs)'].values[:-1], sim['pThr_jnk3'][t_exp_mask] / jnk3_initial_value, color='blue', label='pJNK3 by MKK7 sim')
    plt.errorbar(exp_data['Time (secs)'].values[:-1], exp_data['pThr_arrestin_avg'].values[:-1],
                 exp_data['pThr_arrestin_std'].values[:-1],
                 linestyle='None', marker='o', capsize=5, color='blue', label='pJNK3 by MKK7 exp')

    param_values[arrestin_idx] = 0
    sim2 = solver.run(param_values=param_values).all

    plt.plot(exp_data['Time (secs)'].values[:-1], sim2['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value, color='black', label='pJNK3 by MKK4 sim')
    plt.errorbar(exp_data['Time (secs)'].values[:-1], exp_data['pTyr_noarrestin_avg'].values[:-1],
                 exp_data['pTyr_noarrestin_std'].values[:-1],
                 linestyle='None', marker='o', capsize=5, color='black', label='pJNK3 by MKK4 no Arr exp')
    plt.plot(exp_data['Time (secs)'].values[:-1], sim2['pThr_jnk3'][t_exp_mask] / jnk3_initial_value, color='green', label='pJNK3 by MKK7 sim')
    plt.errorbar(exp_data['Time (secs)'].values[:-1], exp_data['pThr_noarrestin_avg'].values[:-1],
                 exp_data['pThr_noarrestin_std'].values[:-1],
                 linestyle='None', marker='o', capsize=5, color='green', label='pJNK3 by MKK7 no Arr exp')

    plt.xlabel('Arrestin (microM)')
    plt.ylabel('pJNK3 (microM)')
    plt.legend()
    plt.savefig('jnk3_noASK1_trained_new7.png')
    plt.show()

def likelihood(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    pars = np.copy(param_values)

    sim = solver.run(param_values=pars).all

    e_mkk4 = np.sum((exp_data['pTyr_arrestin_avg'].values[:-1] - sim['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                    (2 * exp_data['pTyr_arrestin_std'].values[:-1])) / len(exp_data['pTyr_arrestin_std'].values[:-1])
    e_mkk7 = np.sum((exp_data['pThr_arrestin_avg'].values[:-1] - sim['pThr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                    (2 * exp_data['pThr_arrestin_std'].values[:-1])) / len(exp_data['pThr_arrestin_std'].values[:-1])
    error1 = e_mkk4 + e_mkk7

    # No arrestin experiments
    pars[arrestin_idx] = 0
    sim2 = solver.run(param_values=pars).all
    e2_mkk4 = np.sum((exp_data['pTyr_noarrestin_avg'].values[:-1] - sim2['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                    (2 * exp_data['pTyr_noarrestin_std'].values[:-1])) / len(exp_data['pTyr_noarrestin_std'].values[:-1])
    e2_mkk7 = np.sum((exp_data['pThr_noarrestin_avg'].values[:-1] - sim2['pThr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                    (2 * exp_data['pThr_noarrestin_std'].values[:-1])) / len(exp_data['pThr_noarrestin_std'].values[:-1])
    error2 = e2_mkk4 + e2_mkk7
    error = error1 + error2
    return error,

new_nominal = np.load('jnk3_noASK1_calibrated_pars_new7.npy')

def run_example():
    pso = PSO(save_sampled=False, verbose=True, num_proc=4)
    pso.set_cost_function(likelihood)
    pso.set_start_position(xnominal)
    pso.set_bounds(2)
    pso.set_speed(-.25, .25)
    pso.run(25, 100)
    print ('aca', pso.best.fitness.values)
    display(pso.best)
    np.save('jnk3_noASK1_calibrated_pars_new7', pso.best)


def run_example_multiple():
    best_pars = np.zeros((100, len(model.parameters)))
    counter = 0
    for i in range(100):
        pso = PSO(save_sampled=False, verbose=False, num_proc=4)
        pso.set_cost_function(likelihood)
        nominal_random = new_nominal + np.random.uniform(-1, 1, len(new_nominal))
        pso.set_start_position(nominal_random)
        pso.set_bounds(2.5)
        pso.set_speed(-.25, .25)
        pso.run(25, 100)
        if pso.best.fitness.values[0] < 0.041:
            Y = np.copy(pso.best)
            param_values[rates_of_interest_mask] = 10 ** Y
            best_pars[counter] = param_values
            counter += 1
        print (i, counter)

        # display(pso.best)
    np.save('jnk3_noASK1_ncalibrated_pars', best_pars)

if __name__ == '__main__':
    # run_example()
    run_example_multiple()