from jnk3_no_ask1 import model
import numpy as np
import matplotlib.pyplot as plt
from pysb.simulator import ScipyOdeSimulator

param_values = np.array([p.value for p in model.parameters])
idx_pars_calibrate = [3, 21, 23, 25, 27, 29, 32, 33, 34, 35, 36, 37,  39, 41]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]
pars = np.load('jnk3_noASK1_calibrated_pars_new3.npy')
param_values[rates_of_interest_mask] = 10 ** pars

mkk4_ic_idx = 43
mkk7_ic_idx = 44

tspan = np.linspace(0, 60, 100)

pars1 = np.copy(param_values)
pars1[mkk4_ic_idx] = pars1[mkk4_ic_idx] * 0.5
pars2 = np.copy(param_values)
pars2[mkk7_ic_idx] = pars2[mkk7_ic_idx] * 0.5
all_pars = [param_values, pars1,  pars2]

sims = ScipyOdeSimulator(model=model, tspan=tspan, param_values=all_pars).run().all
labels = ['WT', 'MKK4 KD', 'MKK7 KD']
for idx, sim in enumerate(sims):
    # plt.plot(tspan, sim['all_jnk3'], label=labels[idx])
    plt.plot(tspan, sim['all_jnk3'], label=labels[idx])
    # plt.plot(tspan, sim['all_jnk3'], label=labels[idx])
plt.legend()
plt.show()