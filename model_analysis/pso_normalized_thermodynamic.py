from model_analysis.jnk3_no_ask1 import model
from simplepso.pso import PSO
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model_analysis.equilibration_function import pre_equilibration

exp_data = pd.read_csv('../data/exp_data_3min.csv') # [0,1] range normalization


## New kds in jnk3 mkk4/7
# idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream
# idx_pars_calibrate = [5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream2
idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 19, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream3

rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

# Index of Initial conditions of Arrestin
arrestin_idx = [44]
jnk3_initial_value = 0.6  # total jnk3
jnk3_initial_idxs = [47, 48, 49]
kcat_idx = [36, 37]

param_values = np.array([p.value for p in model.parameters])
nominal_values = np.array([p.value for p in model.parameters])
xnominal = np.log10(nominal_values[rates_of_interest_mask])
lower = xnominal - 2
upper = xnominal + 2
# lower[0] = xnominal[0] - np.log10(534)
# upper[0] = xnominal[0] + np.log(534)


tspan = np.linspace(0, exp_data['Time (secs)'].values[-1], 181)
t_exp_mask = [idx in exp_data['Time (secs)'].values[:] for idx in tspan]

solver = ScipyOdeSimulator(model, tspan=tspan)


def display(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    pars1 = np.copy(param_values)
    pars2 = np.copy(param_values)

    # Pre-equilibration
    time_eq = np.linspace(0, 100, 100)
    pars_eq1 = np.copy(param_values)
    pars_eq2 = np.copy(param_values)

    pars_eq2[arrestin_idx] = 0
    pars_eq2[jnk3_initial_idxs] = [0.5958, 0, 0.0042]

    all_pars = np.stack((pars_eq1, pars_eq2))
    all_pars[:, kcat_idx] = 0  # Setting catalytic reactions to zero for pre-equilibration
    eq_conc = pre_equilibration(model, time_eq, all_pars)[1]

    # Simulating models with initials from pre-equilibration and parameters for condition with/without arrestin
    pars2[arrestin_idx] = 0
    pars2[jnk3_initial_idxs] = [0.5958, 0, 0.0042]
    sim = solver.run(param_values=[pars1, pars2], initials=eq_conc).all

    plt.plot(tspan, sim[0]['pTyr_jnk3'] / jnk3_initial_value, color='red', label='p(Tyr)JNK3 sim')
    plt.errorbar(exp_data['Time (secs)'].values, exp_data['pTyr_arrestin_avg'].values,
                 exp_data['pTyr_arrestin_std'].values,
                 linestyle='None', marker='o', capsize=5, color='red', label='p(Tyr)JNK3 exp')

    plt.plot(tspan, sim[0]['pThr_jnk3'] / jnk3_initial_value, color='blue', label='p(Thr)JNK3 sim')
    plt.errorbar(exp_data['Time (secs)'].values, exp_data['pThr_arrestin_avg'].values,
                 exp_data['pThr_arrestin_std'].values,
                 linestyle='None', marker='o', capsize=5, color='blue', label='p(Thr)JNK3 exp')

    plt.plot(tspan, sim[0]['all_jnk3'] / jnk3_initial_value, color='cyan', label='ppJNK3 sim')
    plt.errorbar(exp_data['Time (secs)'].values, exp_data['ppjnk3_arrestin_avg'].values,
                 exp_data['ppjnk3_arrestin_std'].values,
                 linestyle='None', marker='o', capsize=5, color='cyan', label='ppJNK3 exp')

    plt.plot(tspan, sim[1]['pTyr_jnk3'] / jnk3_initial_value, color='black', label='p(Tyr)JNK3 -Arr sim')
    plt.errorbar(exp_data['Time (secs)'].values, exp_data['pTyr_noarrestin_avg'].values,
                 exp_data['pTyr_noarrestin_std'].values,
                 linestyle='None', marker='o', capsize=5, color='black', label='p(Tyr)JNK3 -Arr exp')

    plt.plot(tspan, sim[1]['pThr_jnk3'] / jnk3_initial_value, color='green', label='p(Thr)JNK3 -Arr sim')
    plt.errorbar(exp_data['Time (secs)'].values, exp_data['pThr_noarrestin_avg'].values,
                 exp_data['pThr_noarrestin_std'].values,
                 linestyle='None', marker='o', capsize=5, color='green', label='p(Thr)JNK3 -Arr exp')

    plt.plot(tspan, sim[1]['all_jnk3'] / jnk3_initial_value, color='purple', label='ppJNK3 -Arr sim')
    plt.errorbar(exp_data['Time (secs)'].values, exp_data['ppjnk3_noarrestin_avg'].values,
                 exp_data['ppjnk3_noarrestin_std'].values,
                 linestyle='None', marker='o', capsize=5, color='purple', label='ppJNK3 -Arr exp')

    plt.xlabel('Arrestin (microM)')
    plt.ylabel('pJNK3 (microM)')
    # plt.legend()
    plt.savefig('jnk3_noASK1_trained_pso_1h.png')
    plt.show()


def likelihood(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    pars1 = np.copy(param_values)
    pars2 = np.copy(param_values)

    # Pre-equilibration
    time_eq = np.linspace(0, 100, 100)
    pars_eq1 = np.copy(param_values)
    pars_eq2 = np.copy(param_values)

    pars_eq2[arrestin_idx] = 0
    # pars_eq2[jnk3_initial_idxs] = [0.5958, 0, 0.0042]

    all_pars = np.stack((pars_eq1, pars_eq2))
    all_pars[:, kcat_idx] = 0  # Setting catalytic reactions to zero for pre-equilibration
    try:
        eq_conc = pre_equilibration(model, time_eq, all_pars)[1]
    except:
        return np.inf,

    # Simulating models with initials from pre-equilibration and parameters for condition with/without arrestin
    pars2[arrestin_idx] = 0
    # pars2[jnk3_initial_idxs] = [0.5958, 0, 0.0042]
    sim = solver.run(param_values=[pars1, pars2], initials=eq_conc).all

    e_mkk4 = np.sum((exp_data['pTyr_arrestin_avg'].values - sim[0]['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                    (exp_data['pTyr_arrestin_std'].values**2))
    e_mkk7 = np.sum((exp_data['pThr_arrestin_avg'].values - sim[0]['pThr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                    (exp_data['pThr_arrestin_std'].values**2))
    # e_ppjnk3 = np.sum((exp_data['ppjnk3_arrestin_avg'].values - sim[0]['all_jnk3'][t_exp_mask] / jnk3_initial_value) **2 /
    #                   (2 * exp_data['ppjnk3_arrestin_std'].values)) / len(exp_data['ppjnk3_arrestin_std'].values)
    error1 = e_mkk4 + e_mkk7

    e2_mkk4 = np.sum((exp_data['pTyr_noarrestin_avg'].values - sim[1]['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                     (exp_data['pTyr_noarrestin_std'].values**2))
    e2_mkk7 = np.sum((exp_data['pThr_noarrestin_avg'].values - sim[1]['pThr_jnk3'][t_exp_mask] / jnk3_initial_value) ** 2 /
                     (exp_data['pThr_noarrestin_std'].values**2))
    # e2_ppjnk3 = np.sum((exp_data['ppjnk3_noarrestin_avg'].values - sim[1]['all_jnk3'][t_exp_mask] / jnk3_initial_value) **2 /
    #                   (2 * exp_data['ppjnk3_noarrestin_std'].values)) / len(exp_data['ppjnk3_noarrestin_std'].values)

    box1 = (pars1[21]/pars1[20]) * (pars1[23]/pars1[22]) * (1 / (pars1[1] / pars1[0])) * \
           (1 / (pars1[5]/pars1[4]))

    box2 = (pars1[21] / pars1[20]) * (pars1[25] / pars1[24]) * (1 / (pars1[3] / pars1[2])) * \
           (1 / (pars1[27] / pars1[26]))

    box3 = (pars1[13] / pars1[12]) * (pars1[23] / pars1[22]) * (1 / (pars1[1] / pars1[0])) * \
           (1 / (pars1[15] / pars1[14]))

    box4 = (pars1[7] / pars1[6]) * (pars1[25] / pars1[24]) * (1 / (pars1[3] / pars1[2])) * \
           (1 / (pars1[11] / pars1[10]))

    boxes = np.array([box1, box2, box3, box4])
    boxes_error = np.sum((boxes - 1)**2)

    error2 = e2_mkk4 + e2_mkk7
    error = error1 + error2 + boxes_error
    return error,

# new_nominal = np.load('jnk3_noASK1_calibrated_pars_pso_2min_5.npy')

def run_example():
    pso = PSO(save_sampled=False, verbose=True, num_proc=4)
    pso.set_cost_function(likelihood)
    pso.set_start_position(xnominal)
    pso.set_bounds(lower=lower, upper=upper)
    pso.set_speed(-.25, .25)
    pso.run(40, 200)
    display(pso.best)
    np.save('calibrated_pars_pso5', pso.best)


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