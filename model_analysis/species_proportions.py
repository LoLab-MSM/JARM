from tropical.cluster_analysis import AnalysisCluster as AC
from jnk3_no_ask1 import model
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from equilibration_function import pre_equilibration

tspan = np.linspace(0, 120, 120)

idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43]

rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]
param_values = np.array([p.value for p in model.parameters])
pars_fitted = np.load('most_likely_par_100000.npy')
param_values[rates_of_interest_mask] = 10 ** pars_fitted
# param_values[44] = 50

kcat_idx = [36, 37]
time_eq = np.linspace(0, 100, 100)
pars_eq1 = np.copy(param_values)
pars_eq1[kcat_idx] = 0
eq_conc = pre_equilibration(model, time_eq, pars_eq1)[1]
sim = ScipyOdeSimulator(model, tspan=tspan).run(param_values=[param_values, param_values],
                                                initials=[eq_conc[0], eq_conc[0]])

# import matplotlib.pyplot as plt
# #
# # plt.plot(tspan, sim.all[0]['__s19'], label='s19')
# # plt.plot(tspan, sim.all[0]['__s10'], label='s10')
# plt.plot(tspan, sim.all[0]['__s2'], label='s2')
# # plt.plot(tspan, sim.all[0]['__s21'], label='s21')
# #
# # plt.legend()
# plt.show()

a = AC(model, sim, clusters=None)

jnk3 = model.monomers[3]
mkk4 = model.monomers['MKK4']
mkk7 = model.monomers['MKK7']


a.hist_avg_sps(jnk3, fig_name='bar_jnk3_proportion', type_fig='entropy')
# a.hist_avg_rxns(jnk3, fig_name='ent2_jnk3_proportion', type_fig='entropy')