from jnk3_no_ask1 import model
from simplepso.pso import PSO
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from equilibration_function import pre_equilibration

# exp_data = pd.read_csv('../data/exp_data_arrestin_50_correction.csv') # 50% correction
exp_data = pd.read_csv('../data/exp_data_arrestin_normalization_1h_138max.csv') # [0,1] range normalization


# idx_pars_calibrate = [3, 21, 23, 25, 27, 29, 32, 33, 34, 35, 36, 37,  39, 41]
# Kcat is same for phosphorylation with and without arrestin
## New kds in jnk3 mkk4/7
# idx_pars_calibrate = [3, 21, 23, 25, 32, 33, 36, 37]

idx_pars_calibrate = [3, 7, 9, 11, 13, 15, 19, 21, 23, 25, 27, 29, 32, 33, 36, 37, 39, 41]

# idx_pars_calibrate = [idx for idx, par in enumerate(model.parameters)
#                       if par not in list(model.parameters_unused()) + list(model.parameters_initial_conditions())]


rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

# Index of Initial conditions of Arrestin
arrestin_idx = [42]
jnk3_initial_value = 0.6 # total jnk3
jnk3_initial_idxs = [45, 46, 47]

param_values = np.array([p.value for p in model.parameters])
nominal_values = np.array([p.value for p in model.parameters])
xnominal = np.log10(nominal_values[rates_of_interest_mask])
lower = xnominal - 2
upper = xnominal + 2
lower[0] = xnominal[0] - np.log10(534)
upper[0] = xnominal[0] + np.log(534)
lower[1] = xnominal[1] - np.log10(0.4)
upper[1] = xnominal[1] + np.log(0.4)
lower[2] = xnominal[2] - np.log10(0.6)
upper[2] = xnominal[2] + np.log(0.6)
lower[3] = xnominal[3] - np.log10(3.8)
upper[3] = xnominal[3] + np.log(3.8)
lower[4] = xnominal[4] - np.log10(7.8)
upper[4] = xnominal[4] + np.log(7.8)
lower[5] = xnominal[5] - np.log10(242)
upper[5] = xnominal[5] + np.log(242)
lower[6] = xnominal[6] - np.log10(0.6)
upper[6] = xnominal[6] + np.log(0.6)
lower[7] = xnominal[7] - np.log10(20)
upper[7] = xnominal[7] + np.log(20)
lower[8] = xnominal[8] - np.log10(20)
upper[8] = xnominal[8] + np.log(20)
lower[9] = xnominal[9] - np.log10(0.6)
upper[9] = xnominal[9] + np.log(0.6)
lower[10] = xnominal[10] - np.log10(20)
upper[10] = xnominal[10] + np.log(20)
lower[11] = xnominal[11] - np.log10(20)
upper[11] = xnominal[11] + np.log(20)
lower[16] = xnominal[16] - np.log10(20)
upper[16] = xnominal[16] + np.log(20)
lower[17] = xnominal[17] - np.log10(20)
upper[17] = xnominal[17] + np.log(20)

ignore = 1

tspan = np.linspace(0, exp_data['Time (secs)'].values[-(ignore+1)], 121)
t_exp_mask = [idx in exp_data['Time (secs)'].values[:-1] for idx in tspan]
solver = ScipyOdeSimulator(model, tspan=tspan)


def display(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    sim = solver.run(param_values=param_values).all

    plt.plot(tspan, sim['pTyr_jnk3'] / jnk3_initial_value, color='red', label='p(Tyr)JNK3 sim')
    plt.errorbar(exp_data['Time (secs)'].values[:-ignore], exp_data['pTyr_arrestin_avg'].values[:-ignore],
                 exp_data['pTyr_arrestin_std'].values[:-ignore],
                 linestyle='None', marker='o', capsize=5, color='red', label='p(Tyr)JNK3 exp')
    plt.plot(tspan, sim['pThr_jnk3'] / jnk3_initial_value, color='blue', label='p(Thr)JNK3 sim')
    plt.errorbar(exp_data['Time (secs)'].values[:-ignore], exp_data['pThr_arrestin_avg'].values[:-ignore],
                 exp_data['pThr_arrestin_std'].values[:-ignore],
                 linestyle='None', marker='o', capsize=5, color='blue', label='p(Thr)JNK3 exp')

    param_values[arrestin_idx] = 0
    param_values[jnk3_initial_idxs] = [0.492, 0.108, 0]
    sim2 = solver.run(param_values=param_values).all

    plt.plot(tspan, sim2['pTyr_jnk3'] / jnk3_initial_value, color='black', label='p(Tyr)JNK3 -Arr sim')
    plt.errorbar(exp_data['Time (secs)'].values[:-ignore], exp_data['pTyr_noarrestin_avg'].values[:-ignore],
                 exp_data['pTyr_noarrestin_std'].values[:-ignore],
                 linestyle='None', marker='o', capsize=5, color='black', label='p(Tyr)JNK3 -Arr exp')
    plt.plot(tspan, sim2['pThr_jnk3'] / jnk3_initial_value, color='green', label='p(Thr)JNK3 -Arr sim')
    plt.errorbar(exp_data['Time (secs)'].values[:-ignore], exp_data['pThr_noarrestin_avg'].values[:-ignore],
                 exp_data['pThr_noarrestin_std'].values[:-ignore],
                 linestyle='None', marker='o', capsize=5, color='green', label='p(Thr)JNK3 -Arr exp')

    plt.xlabel('Arrestin (microM)')
    plt.ylabel('pJNK3 (microM)')
    plt.legend()
    plt.savefig('jnk3_noASK1_trained_pso25_1h.png')
    plt.show()


def likelihood(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    pars = np.copy(param_values)

    # Pre-equilibration
    time_eq = np.linspace(0, 60, 60)
    pars_eq1 = np.copy(pars)
    eq_conc1 = pre_equilibration(model, time_eq, pars_eq1)[1]

    sim = solver.run(param_values=pars, initials=eq_conc1).all

    e_mkk4 = np.sum((exp_data['pTyr_arrestin_avg'].values[:-ignore] - sim['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                    (2 * exp_data['pTyr_arrestin_std'].values[:-ignore])) / len(exp_data['pTyr_arrestin_std'].values[:-ignore])
    e_mkk7 = np.sum((exp_data['pThr_arrestin_avg'].values[:-ignore] - sim['pThr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                    (2 * exp_data['pThr_arrestin_std'].values[:-ignore])) / len(exp_data['pThr_arrestin_std'].values[:-ignore])
    error1 = e_mkk4 + e_mkk7

    # No arrestin experiments
    # Pre equilibration
    pars_eq1[arrestin_idx] = 0
    pars_eq1[jnk3_initial_idxs] = [0.492, 0.108, 0]
    eq_conc2 = pre_equilibration(model, time_eq, pars_eq1)[1]

    pars[arrestin_idx] = 0
    pars[jnk3_initial_idxs] = [0.492, 0.108, 0]
    sim2 = solver.run(param_values=pars, initials=eq_conc2).all
    e2_mkk4 = np.sum((exp_data['pTyr_noarrestin_avg'].values[:-ignore] - sim2['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                    (2 * exp_data['pTyr_noarrestin_std'].values[:-ignore])) / len(exp_data['pTyr_noarrestin_std'].values[:-ignore])
    e2_mkk7 = np.sum((exp_data['pThr_noarrestin_avg'].values[:-ignore] - sim2['pThr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                    (2 * exp_data['pThr_noarrestin_std'].values[:-ignore])) / len(exp_data['pThr_noarrestin_std'].values[:-ignore])
    error2 = e2_mkk4 + e2_mkk7
    error = error1 + error2
    return error,

new_nominal = np.load('jnk3_noASK1_calibrated_pars_pso24_1h.npy')

def run_example():
    pso = PSO(save_sampled=False, verbose=True, num_proc=4)
    pso.set_cost_function(likelihood)
    pso.set_start_position(new_nominal)
    pso.set_bounds(lower=lower, upper=upper)
    pso.set_speed(-.25, .25)
    pso.run(25, 100)
    print ('aca', pso.best.fitness.values)
    display(pso.best)
    np.save('jnk3_noASK1_calibrated_pars_pso25_1h', pso.best)


def run_example_multiple():
    best_pars = np.zeros((100, len(model.parameters)))
    counter = 0
    for i in range(100):
        pso = PSO(save_sampled=False, verbose=False, num_proc=4)
        pso.set_cost_function(likelihood)
        nominal_random = xnominal + np.random.uniform(-1, 1, len(xnominal))
        pso.set_start_position(nominal_random)
        pso.set_bounds(2.5)
        pso.set_speed(-.25, .25)
        pso.run(25, 100)
        if pso.best.fitness.values[0] < 0.066:
            Y = np.copy(pso.best)
            param_values[rates_of_interest_mask] = 10 ** Y
            best_pars[counter] = param_values
            counter += 1
        print (i, counter)

        # display(pso.best)
    np.save('jnk3_noASK1_ncalibrated_pars_1h', best_pars)

if __name__ == '__main__':
    run_example()
    # run_example_multiple()