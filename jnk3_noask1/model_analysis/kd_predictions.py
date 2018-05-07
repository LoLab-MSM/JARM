from jnk3_no_ask1 import model
import numpy as np
import matplotlib.pyplot as plt
from pysb.simulator import ScipyOdeSimulator
from equilibration_function import pre_equilibration

param_values = np.array([p.value for p in model.parameters])
#New kds in jnk3 mkk4/7
idx_pars_calibrate = [1, 15, 17, 19, 24, 25, 26, 27]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]
pars = np.load('jnk3_noASK1_calibrated_pars_pso_1h.npy')
param_values[rates_of_interest_mask] = 10 ** pars

mkk4_ic_idx = 33
mkk7_ic_idx = 34

tspan = np.linspace(0, 60, 100)

pars1 = np.copy(param_values)
pars1[mkk4_ic_idx] = pars1[mkk4_ic_idx] * 0.5
pars2 = np.copy(param_values)
pars2[mkk7_ic_idx] = pars2[mkk7_ic_idx] * 0.5
all_pars = [param_values, pars1,  pars2]


# Pre-equilibration
# Index of Initial conditions of Arrestin
time_eq = np.linspace(0, 30, 30)
pars_eq = np.copy(param_values)
pars_eq1 = np.copy(pars1)
pars_eq2 = np.copy(pars2)

all_pars_eq = np.stack((pars_eq, pars_eq1, pars_eq2))
all_pars_eq[:, [24, 25]] = 0  # Setting catalytic reactions to zero for pre-equilibration
eq_conc = pre_equilibration(model, time_eq, all_pars_eq)[1]


sims = ScipyOdeSimulator(model=model, tspan=tspan).run(param_values=all_pars, initials=eq_conc).all
labels = ['WT', 'MKK4 KD', 'MKK7 KD']
for idx, sim in enumerate(sims):
    # plt.plot(tspan, sim['all_jnk3'], label=labels[idx])
    plt.plot(tspan, sim['all_jnk3'], label=labels[idx])
    # plt.plot(tspan, sim['all_jnk3'], label=labels[idx])
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel(r'Concentration [$\mu$M]')
plt.savefig('mkk4_mkk7_knockdows_predictions_test.pdf', format='pdf')