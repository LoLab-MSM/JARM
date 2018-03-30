from jnk3_noask1.jnk3_no_ask1 import model
from simplepso.pso import PSO
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mkk4_data = np.loadtxt('../data/mkk4_data.csv', delimiter=',')
sd_mkk4_data = np.loadtxt('../data/sd_mkk4_data.csv', delimiter=',')
mkk7_data = np.loadtxt('../data/mkk7_data.csv', delimiter=',')
sd_mkk7_data = np.loadtxt('../data/sd_mkk7_data.csv', delimiter=',')

exp_data = pd.read_csv('../data/exp_data_arrestin_noarrestin.csv')

idx_pars_calibrate = [3, 9, 11, 21, 23, 25, 27, 29, 32, 33, 34, 35, 36, 37, 39, 41]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

# Initial conditions of mkk4, mkk7, uJNK3 respectively
initials_experiment_mkk4 = [0.05, 0, 0.5]
initials_experiment_mkk7 = [0, 0.05, 0.5]
initials_exps_idxs = [43, 44, 45]
arrestin_idx = [42]

param_values = np.array([p.value for p in model.parameters])
nominal_values = np.array([p.value for p in model.parameters])
xnominal = np.log10(nominal_values[rates_of_interest_mask])
tspan = np.linspace(0, 10, 20)
tspan2 = exp_data['Time (secs)'].values
solver = ScipyOdeSimulator(model, tspan=tspan)
solver2 = ScipyOdeSimulator(model, tspan=tspan2)


def display(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    jnk3_mkk4_pars = [0] * 13
    jnk3_mkk7_pars = [0] * 13

    param_values[initials_exps_idxs] = initials_experiment_mkk4
    for i, arr_conc in enumerate(mkk4_data[:, 0]):
        param_values[arrestin_idx] = arr_conc
        jnk3_mkk4_pars[i] = np.copy(param_values)

    param_values[initials_exps_idxs] = initials_experiment_mkk7
    for i, arr_conc in enumerate(mkk7_data[:, 0]):
        param_values[arrestin_idx] = arr_conc
        jnk3_mkk7_pars[i] = np.copy(param_values)

    all_pars = jnk3_mkk4_pars + jnk3_mkk7_pars

    sims = solver2.run(param_values=all_pars).all
    jnk3_mkk4_sim = [sim['pTyr_jnk3'][-1] for sim in sims[:13]]
    jnk3_mkk7_sim = [sim['pThr_jnk3'][-1] for sim in sims[13:26]]

    plt.semilogx(mkk4_data[:, 0], jnk3_mkk4_sim, 'x', color='red', label='pJNK3 by MKK4 sim')
    plt.errorbar(mkk4_data[:, 0], mkk4_data[:, 1], sd_mkk4_data[:, 1], linestyle='None', marker='o',
                 capsize=5, color='red', label='pJNK3 by MKK4 exp')
    plt.semilogx(mkk7_data[:, 0], jnk3_mkk7_sim, 'x', color='blue', label='pJNK3 by MKK7 sim')
    plt.errorbar(mkk7_data[:, 0], mkk7_data[:, 1], sd_mkk7_data[:, 1], linestyle='None', marker='o',
                 capsize=5, color='blue', label='pJNK3 by MKK7 exp')
    plt.xlabel('Arrestin (microM)')
    plt.ylabel('pJNK3 (microM)')
    plt.legend()
    plt.savefig('jnk3_noASK1_ic_ss_trained2.png')
    plt.show()

def display1(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    jnk3_mkk4_pars = [0] * 13
    jnk3_mkk7_pars = [0] * 13

    param_values[initials_exps_idxs] = initials_experiment_mkk4
    for i, arr_conc in enumerate(mkk4_data[:, 0]):
        param_values[arrestin_idx] = arr_conc
        jnk3_mkk4_pars[i] = np.copy(param_values)

    param_values[initials_exps_idxs] = initials_experiment_mkk7
    for i, arr_conc in enumerate(mkk7_data[:, 0]):
        param_values[arrestin_idx] = arr_conc
        jnk3_mkk7_pars[i] = np.copy(param_values)

    all_pars = jnk3_mkk4_pars + jnk3_mkk7_pars

    sims = solver.run(param_values=all_pars).all
    jnk3_mkk4_sim = [sim['pTyr_jnk3'][-1] for sim in sims[:13]]
    jnk3_mkk7_sim = [sim['pThr_jnk3'][-1] for sim in sims[13:26]]

    plt.semilogx(mkk4_data[:, 0], jnk3_mkk4_sim, 'x', color='red', label='pJNK3 by MKK4 sim')
    plt.errorbar(mkk4_data[:, 0], mkk4_data[:, 1], sd_mkk4_data[:, 1], linestyle='None', marker='o',
                 capsize=5, color='red', label='pJNK3 by MKK4 exp')
    plt.semilogx(mkk7_data[:, 0], jnk3_mkk7_sim, 'x', color='blue', label='pJNK3 by MKK7 sim')
    plt.errorbar(mkk7_data[:, 0], mkk7_data[:, 1], sd_mkk7_data[:, 1], linestyle='None', marker='o',
                 capsize=5, color='blue', label='pJNK3 by MKK7 exp')
    plt.xlabel('Time (s)')
    plt.ylabel('pJNK3 (microM)')
    plt.legend()
    plt.savefig('jnk3_noASK1_alldata_trained.png')
    plt.show()


def likelihood(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y
    pars = np.copy(param_values)

    jnk3_mkk4_pars = [0] * 13
    jnk3_mkk7_pars = [0] * 13

    param_values[initials_exps_idxs] = initials_experiment_mkk4
    for i, arr_conc in enumerate(mkk4_data[:, 0]):
        param_values[arrestin_idx] = arr_conc
        jnk3_mkk4_pars[i] = np.copy(param_values)

    param_values[initials_exps_idxs] = initials_experiment_mkk7
    for i, arr_conc in enumerate(mkk7_data[:, 0]):
        param_values[arrestin_idx] = arr_conc
        jnk3_mkk7_pars[i] = np.copy(param_values)

    all_pars = jnk3_mkk4_pars + jnk3_mkk7_pars

    sims = solver.run(param_values=all_pars).all
    jnk3_mkk4_sim = [sim['pTyr_jnk3'][-1] for sim in sims[:13]]
    jnk3_mkk7_sim = [sim['pThr_jnk3'][-1] for sim in sims[13:26]]

    e_mkk4 = np.sum((mkk4_data[:, 1] - jnk3_mkk4_sim) ** 2 / (2 * sd_mkk4_data[:, 1])) / len(sd_mkk4_data[:, 1])
    e_mkk7 = np.sum((mkk7_data[:, 1] - jnk3_mkk7_sim) ** 2 / (2 * sd_mkk7_data[:, 1])) / len(sd_mkk7_data[:, 1])

    pars[initials_exps_idxs] = [0.2, 0.2, 2]
    pars[arrestin_idx] = 20
    sim = solver2.run(param_values=pars).all
    e1_mkk4 = np.sum((exp_data['pTyr_arrestin_avg'].values - sim['pTyr_jnk3']) ** 2 /
                    (2 * exp_data['pTyr_arrestin_std'].values)) / len(exp_data['pTyr_arrestin_std'].values)
    e1_mkk7 = np.sum((exp_data['pThr_arrestin_avg'].values - sim['pThr_jnk3']) ** 2 /
                    (2 * exp_data['pThr_arrestin_std'].values)) / len(exp_data['pThr_arrestin_std'].values)

    # No arrestin experiments
    pars[arrestin_idx] = 0
    sim2 = solver2.run(param_values=pars).all
    e2_mkk4 = np.sum((exp_data['pTyr_noarrestin_avg'].values - sim2['pTyr_jnk3']) ** 2 /
                    (2 * exp_data['pTyr_noarrestin_std'].values)) / len(exp_data['pTyr_noarrestin_std'].values)
    e2_mkk7 = np.sum((exp_data['pThr_noarrestin_avg'].values - sim2['pThr_jnk3']) ** 2 /
                    (2 * exp_data['pThr_noarrestin_std'].values)) / len(exp_data['pThr_noarrestin_std'].values)

    error = e_mkk4 + e_mkk7 + e1_mkk4 + e1_mkk7 + e2_mkk4 + e2_mkk7
    return error,

new_nominal = np.load('jnk3_noASK1_ic_ss_pars.npy')

def run_example():
    pso = PSO(save_sampled=False, verbose=True, num_proc=4)
    pso.set_cost_function(likelihood)
    pso.set_start_position(xnominal)
    pso.set_bounds(2)
    pso.set_speed(-.25, .25)
    pso.run(50, 100)
    np.save('jnk3_noASK1_alldata_pars', pso.best)
    display(pso.best)
    display1(pso.best)

if __name__ == '__main__':
    run_example()