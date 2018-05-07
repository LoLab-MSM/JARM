# coding=utf-8
from jnk3_no_ask1_final import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
import matplotlib.pyplot as plt
from equilibration_function import pre_equilibration

#New kds in jnk3 mkk4/7
idx_pars_calibrate = [1, 15, 17, 19, 24, 25, 26, 27]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

calibrated_pars = np.load('jnk3_noASK1_calibrated_pars_pso_1h.npy')
param_values = np.array([p.value for p in model.parameters])

par_set_calibrated = np.copy(param_values)
par_set_calibrated[rates_of_interest_mask] = 10 ** calibrated_pars

tspan = np.linspace(0, 10, 100)

n_conditions = 100
max_arrestin = 40
arrestin_initials = np.linspace(0, max_arrestin, n_conditions)
par_clus1 = par_set_calibrated

repeated_parameter_values = np.tile(par_clus1, (n_conditions, 1))
repeated_parameter_values[:, 32] = arrestin_initials
np.save('arrestin_diff_IC_par0.npy', repeated_parameter_values)

time_eq = np.linspace(0, 30, 30)
pars_ic_eq = np.copy(repeated_parameter_values)
pars_ic_eq[:, [24, 25]] = 0  # Setting catalytic reactions to zero for pre-equilibration
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

plt.savefig('plot_varying_arrestin_pars_pso_1h_test.pdf', format='pdf')
# plt.show()