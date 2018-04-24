# coding=utf-8
from jnk3_no_ask1 import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
import seaborn as sn
import matplotlib.pyplot as plt
from equilibration_function import pre_equilibration

# idx_pars_calibrate = [3, 21, 23, 25, 27, 29, 32, 33, 36, 37,  39, 41]
#New kds in jnk3 mkk4/7
idx_pars_calibrate = [3, 21, 23, 25, 32, 33, 36, 37]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

calibrated_pars = np.load('pydream_most_likely_par.npy')
# calibrated_pars = np.load('pydream_most_likely_par.npy')
param_values = np.array([p.value for p in model.parameters])

jnk3_initial_idxs = [45, 46, 47]

par_set_calibrated = np.copy(param_values)
par_set_calibrated[rates_of_interest_mask] = 10 ** calibrated_pars

tspan = np.linspace(0, 60, 100)

# Pre-equilibration
time_eq = np.linspace(0, 30, 30)
pars_eq_nc = np.array([par.value for par in model.parameters])
pars_eq_nc[[16, 17, 32, 33, 34, 35]] = 0
eq_conc_nc = pre_equilibration(model, time_eq, pars_eq_nc)[1]

sim1 = ScipyOdeSimulator(model, tspan, initials=eq_conc_nc).run().all

# Pre-equilibration
time_eq = np.linspace(0, 30, 30)
# par_set_calibrated[44] = 0.3
# par_set_calibrated[42] = 0.4
pars_eq = np.copy(par_set_calibrated)
pars_eq[[16, 17, 32, 33, 34, 35]] = 0
eq_conc1 = pre_equilibration(model, time_eq, pars_eq)[1]

sim2 = ScipyOdeSimulator(model, tspan, param_values=par_set_calibrated, initials=eq_conc1).run().all

arrestin_idx = 42

# No arrestin experiments
# Pre equilibration
pars_eq[arrestin_idx] = 0
pars_eq[jnk3_initial_idxs] = [0.492, 0.108, 0]
eq_conc2 = pre_equilibration(model, time_eq, pars_eq)[1]

par_set_calibrated[arrestin_idx] = 0
par_set_calibrated[jnk3_initial_idxs] = [0.492, 0.108, 0]
sim3 = ScipyOdeSimulator(model, tspan, param_values=par_set_calibrated, initials=eq_conc2).run().all


def plot_trajectories_nocalibrated_model():
    linestyle = 'dashed'
    size = 8
    loc = (1.01, 0.8)
    frameon = False
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(2, sharex=True)
    # axarr[0].set_title('Sharing X axis')
    axarr[0].plot(tspan, sim1['__s13'], linestyle=linestyle, color='#0072B2', label='Arrestin-3:MKK4:p(Tyr)JNK3')
    axarr[0].plot(tspan, sim1['__s14'], color='#0072B2', label='Arrestin-3:MKK7:p(Thr)JNK3')
    axarr[0].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)

    # axarr[2].plot(tspan, sim2['pTyr_jnk3'], linestyle=linestyle, color='#D55E00')
    # axarr[2].plot(tspan, sim2['pThr_jnk3'], color='#D55E00')

    axarr[1].plot(tspan, sim1['__s19'], linestyle=linestyle, color='#CC79A7', label='Arrestin-3:MKK4:p(Thr)JNK3')
    axarr[1].plot(tspan, sim1['__s20'], color='#CC79A7', label='Arrestin-3:MKK7:p(Tyr)JNK3')
    axarr[1].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)
    axarr[1].set_xlabel('Time (s)')
    f.text(0.01, 0.5, r'Concentration [$\mu$M]', ha='center', va='center', rotation='vertical')
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.subplots_adjust(right=0.7)
    plt.savefig('model_no_calibrated_trajectories_pydream.pdf', format='pdf', bbox_inches="tight")
#
# plt.figure(2)
# plt.plot(tspan, sim2['__s13'], label='Arr%upJNK3%MKK4')
# plt.plot(tspan, sim2['__s14'], label='Arr%puJNK3%MKK7')
# plt.legend()
# plt.xlabel('Time(s)')
# plt.ylabel('Concentration mM')
# plt.show()


def plot_trajectories_calibrated_model():
    linestyle = 'dashed'
    size = 8
    loc = (1.01, 0)
    frameon = False
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(7, sharex=True)
    axarr[0].plot(tspan, sim2['__s6'], linestyle=linestyle, color='#E69F00', label="Arrestin-3:MKK4")
    axarr[0].plot(tspan, sim2['__s7'], color='#E69F00', label="Arrestin-3:MKK7")
    axarr[0].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)
    # axarr[0].set_title('Sharing X axis')
    axarr[1].plot(tspan, sim2['__s18'], linestyle=linestyle, color='#0072B2', label='Arrestin-3:MKK4:p(Tyr)JNK3')
    axarr[1].plot(tspan, sim2['__s21'], color='#0072B2', label='Arrestin-3:MKK7:p(Thr)JNK3')
    axarr[1].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)

    # axarr[2].plot(tspan, sim2['pTyr_jnk3'], linestyle=linestyle, color='#D55E00')
    # axarr[2].plot(tspan, sim2['pThr_jnk3'], color='#D55E00')

    axarr[2].plot(tspan, sim2['__s19'], linestyle=linestyle, color='#CC79A7', label='Arrestin-3:MKK4:p(Thr)JNK3')
    axarr[2].plot(tspan, sim2['__s22'], color='#CC79A7', label='Arrestin-3:MKK7:p(Tyr)JNK3')
    axarr[2].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)
    # axarr[3].plot(tspan, sim1['__s19'], linestyle=linestyle, color='#000000')
    # axarr[3].plot(tspan, sim1['__s20'], color='#000000')
    axarr[3].plot(tspan, sim2['__s25'], linestyle=linestyle, color='#009E73', label='Arrestin-3:MKK4:p(Tyr-Thr)JNK3')
    axarr[3].plot(tspan, sim2['__s26'], color='#009E73', label='Arrestin-3:MKK7:p(Tyr-Thr)JNK3')
    axarr[3].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)

    axarr[4].plot(tspan, sim2['__s13'], linestyle=linestyle, color='#F0E442', label='MKK4:p(Tyr)JNK3')
    axarr[4].plot(tspan, sim2['__s16'], color='#F0E442', label='MKK7:p(Thr)JNK3')
    axarr[4].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)

    axarr[5].plot(tspan, sim2['__s12'], linestyle=linestyle, color='#D55E00', label='MKK4:p(Thr)JNK3')
    axarr[5].plot(tspan, sim2['__s25'], color='#D55E00', label='MKK7:(Tyr)JNK3')
    axarr[5].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)

    axarr[6].plot(tspan, sim2['__s23'], linestyle=linestyle, color='#56B4E9', label='MKK4:p(Tyr-Thr)JNK3')
    axarr[6].plot(tspan, sim2['__s24'], color='#56B4E9', label='MKK7:p(Tyr-Thr)JNK3')
    axarr[6].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)

    axarr[6].set_xlabel('Time (s)')
    f.text(0.01, 0.5, r'Concentration [$\mu$M]', ha='center', va='center', rotation='vertical')
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.subplots_adjust(right=0.7)
    plt.subplots_adjust(hspace=0.4)

    axarr[0].tick_params(axis='y', which='major', labelsize=8)
    axarr[1].tick_params(axis='y', which='major', labelsize=8)
    axarr[2].tick_params(axis='y', which='major', labelsize=8)
    axarr[3].tick_params(axis='y', which='major', labelsize=8)
    axarr[4].tick_params(axis='y', which='major', labelsize=8)
    axarr[5].tick_params(axis='y', which='major', labelsize=8)
    axarr[6].tick_params(axis='y', which='major', labelsize=8)
    # axarr[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # axarr[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # axarr[3].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # axarr[4].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # axarr[5].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # axarr[6].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.savefig('model_trajectories_calibrated_pydream.pdf', format='pdf', bbox_inches='tight')


def plot_arrestin_noarrestin_ppjnk3():
    plt.plot(tspan, sim2['all_jnk3'], color='r', label='ppJNK3 with Arrestin-3')
    plt.plot(tspan, sim3['all_jnk3'], color='k', label='ppJNK3 no Arrestin-3')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Concentration [$\mu$M]')
    plt.ylim((0, 0.35))
    plt.legend()
    plt.savefig('plot_arrestin_noarrestin_ppjnk3_pydream_arrestin.pdf', format='pdf', bbox_inches='tight')
    print (sim2['all_jnk3'][-1]/sim3['all_jnk3'][-1])



# plot_trajectories_calibrated_model()
# plot_trajectories_nocalibrated_model()
plot_arrestin_noarrestin_ppjnk3()