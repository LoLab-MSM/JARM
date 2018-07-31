# coding=utf-8
from jnk3_no_ask1 import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
import matplotlib.pyplot as plt
from equilibration_function import pre_equilibration

#New kds in jnk3 mkk4/7
# idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream
# idx_pars_calibrate = [5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream2
idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 19, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream3
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

calibrated_pars = np.load('most_likely_par_100000_3.npy')
param_values = np.array([p.value for p in model.parameters])

par_set_calibrated = np.copy(param_values)
par_set_calibrated[rates_of_interest_mask] = 10 ** calibrated_pars

tspan = np.linspace(0, 60, 100)

n_conditions = 500
max_arrestin = 100
arrestin_initials = np.linspace(0, max_arrestin, n_conditions)
arrestin_initials = arrestin_initials
par_clus1 = par_set_calibrated

arrestin_idx = 44
kcat_idx = [36, 37]

repeated_parameter_values = np.tile(par_clus1, (n_conditions, 1))
repeated_parameter_values[:, arrestin_idx] = arrestin_initials
np.save('arrestin_diff_IC_par0.npy', repeated_parameter_values)

time_eq = np.linspace(0, 1000, 100)
pars_ic_eq = np.copy(repeated_parameter_values)
pars_ic_eq[:, kcat_idx] = 0  # Setting catalytic reactions to zero for pre-equilibration
eq_conc = pre_equilibration(model, time_eq, pars_ic_eq)[1]

sim1 = ScipyOdeSimulator(model=model, tspan=tspan).run(param_values=repeated_parameter_values, initials=eq_conc).all

ppjnk3 = np.array([s['all_jnk3'][-1] for s in sim1])
ppjnk3_max_idx = np.argmax(ppjnk3)

plt.plot(arrestin_initials, ppjnk3)
plt.axvline(arrestin_initials[ppjnk3_max_idx], color='r', linestyle = 'dashed', ymax=0.95)
locs, labels = plt.xticks()
locs = np.append(locs, arrestin_initials[ppjnk3_max_idx])
plt.xticks(locs.astype(int))
plt.xlim(0, max_arrestin)
plt.xlabel(r'Arrestin [$\mu$M]')
plt.ylabel(r'doubly phosphorylated JNK3 [$\mu$M]')

plt.savefig('varying_arrestin_3.pdf', format='pdf')
# plt.show()