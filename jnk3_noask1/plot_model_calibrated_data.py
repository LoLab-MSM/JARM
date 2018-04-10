# Importing libraries
from jnk3_no_ask1 import model
import numpy as np
import matplotlib.pyplot as plt
from pysb.simulator import ScipyOdeSimulator
import pandas as pd

# Loading fitted parameters
param_values = np.array([p.value for p in model.parameters])
idx_pars_calibrate = [3, 21, 23, 25, 27, 29, 32, 33, 34, 35, 36, 37,  39, 41]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

fitted_pars = np.load('jnk3_noASK1_calibrated_pars7.npy')
param_values[rates_of_interest_mask] = 10 ** fitted_pars

exp_data = pd.read_csv('../data/exp_data_arrestin_noarrestin.csv')

# tspan0 = exp_data['Time (secs)'].values[:-1]
tspan0 = np.linspace(0, 60, 100)
solver0 = ScipyOdeSimulator(model, tspan=tspan0)
sim = solver0.run(param_values=param_values).all
arrestin_idx = [42]

plt.plot(tspan0, sim['pTyr_jnk3'], color='red')
plt.errorbar(exp_data['Time (secs)'].values[:-1], exp_data['pTyr_arrestin_avg'].values[:-1],
             exp_data['pTyr_arrestin_std'].values[:-1],
             linestyle='None', marker='o', capsize=5, color='red', label='pJNK3 by MKK4 exp')
plt.plot(tspan0, sim['pThr_jnk3'], color='blue')
plt.errorbar(exp_data['Time (secs)'].values[:-1], exp_data['pThr_arrestin_avg'].values[:-1],
             exp_data['pThr_arrestin_std'].values[:-1],
             linestyle='None', marker='o', capsize=5, color='blue', label='pJNK3 by MKK7 exp')

param_values2 = np.copy(param_values)
param_values2[arrestin_idx] = 0
sim2 = solver0.run(param_values=param_values2).all

plt.plot(tspan0, sim2['pTyr_jnk3'], color='black')
plt.errorbar(exp_data['Time (secs)'].values[:-1], exp_data['pTyr_noarrestin_avg'].values[:-1],
             exp_data['pTyr_noarrestin_std'].values[:-1],
             linestyle='None', marker='o', capsize=5, color='black', label='pJNK3 by MKK4 no Arr exp')
plt.plot(tspan0, sim2['pThr_jnk3'], color='green')
plt.errorbar(exp_data['Time (secs)'].values[:-1], exp_data['pThr_noarrestin_avg'].values[:-1],
             exp_data['pThr_noarrestin_std'].values[:-1],
             linestyle='None', marker='o', capsize=5, color='green', label='pJNK3 by MKK7 no Arr exp')

plt.xlabel('Time (s)')
plt.ylabel(r'Concentration [$\mu$M]')
plt.legend()
plt.savefig('model_calibrated_data7.eps', format='eps')