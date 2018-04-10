# coding=utf-8
from jnk3_no_ask1 import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
import seaborn as sn
import matplotlib.pyplot as plt

idx_pars_calibrate = [3, 21, 23, 25, 27, 29, 32, 33, 34, 35, 36, 37,  39, 41]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

calibrated_pars = np.load('jnk3_noASK1_calibrated_pars7.npy')
param_values = np.array([p.value for p in model.parameters])

par_set_calibrated = np.copy(param_values)
par_set_calibrated[rates_of_interest_mask] = 10 ** calibrated_pars

tspan = np.linspace(0, 10, 100)

n_conditions = 100
arrestin_initials = np.linspace(0, 100, n_conditions)
par_clus1 = par_set_calibrated

repeated_parameter_values = np.tile(par_clus1, (n_conditions, 1))
repeated_parameter_values[:, 42] = arrestin_initials
np.save('arrestin_diff_IC_par0.npy', repeated_parameter_values)

sim1 = ScipyOdeSimulator(model=model, tspan=tspan, param_values=repeated_parameter_values).run().all

ppjnk3 = [s['__s27'][-1] for s in sim1]

plt.plot(arrestin_initials, ppjnk3)
plt.xlabel(r'Arrestin [$\mu$M]')
plt.ylabel(r'doubly phosphorylated JNK3 [$\mu$M]')

plt.savefig('plot_varying_arrestin.pdf', format='pdf')
# plt.show()