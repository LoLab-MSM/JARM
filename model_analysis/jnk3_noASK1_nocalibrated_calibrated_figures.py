# coding=utf-8
from jnk3_no_ask1 import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
import seaborn as sn
import matplotlib.pyplot as plt
from equilibration_function import pre_equilibration
import sympy
from collections import OrderedDict

#New kds in jnk3 mkk4/7
idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream
# idx_pars_calibrate = [5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream2
# idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 19, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream3
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

# calibrated_pars = np.load('jnk3_noASK1_calibrated_pars_pso_1h.npy')
calibrated_pars = np.load('most_likely_par_100000_1.npy') # most likely parameter from pydream calibration
param_values = np.array([p.value for p in model.parameters])

jnk3_initial_idxs = [47, 48, 49]
arrestin_idx = 44

par_set_calibrated = np.copy(param_values)
par_set_calibrated[rates_of_interest_mask] = 10 ** calibrated_pars


def plot_trajectories_nocalibrated_model():
    # Pre-equilibration
    time_eq = np.linspace(0, 30, 30)
    pars_eq_nc = np.array([par.value for par in model.parameters])
    pars_eq_nc[[24, 25]] = 0  # Setting catalytic reactions to zero for pre-equilibration
    eq_conc_nc = pre_equilibration(model, time_eq, pars_eq_nc)[1]

    tspan = np.linspace(0, 60, 100)
    sim1 = ScipyOdeSimulator(model, tspan, initials=eq_conc_nc).run().all

    linestyle = 'dashed'
    size = 8
    loc = (1.01, 0.8)
    frameon = False
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(2, sharex=True)
    # axarr[0].set_title('Sharing X axis')
    axarr[0].plot(tspan, sim1['__s18'], linestyle=linestyle, color='#0072B2', label='Arrestin-3:MKK4:p(Tyr)JNK3')
    axarr[0].plot(tspan, sim1['__s21'], color='#0072B2', label='Arrestin-3:MKK7:p(Thr)JNK3')
    axarr[0].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)

    # axarr[2].plot(tspan, sim2['pTyr_jnk3'], linestyle=linestyle, color='#D55E00')
    # axarr[2].plot(tspan, sim2['pThr_jnk3'], color='#D55E00')

    axarr[1].plot(tspan, sim1['__s19'], linestyle=linestyle, color='#CC79A7', label='Arrestin-3:MKK4:p(Thr)JNK3')
    axarr[1].plot(tspan, sim1['__s22'], color='#CC79A7', label='Arrestin-3:MKK7:p(Tyr)JNK3')
    axarr[1].legend(frameon=frameon, loc=loc, prop={'size': size}).get_frame().set_alpha(1)
    axarr[1].set_xlabel('Time (s)')
    f.text(0.01, 0.5, r'Concentration [$\mu$M]', ha='center', va='center', rotation='vertical')
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.subplots_adjust(right=0.7)
    plt.savefig('model_no_calibrated_trajectories.pdf', format='pdf', bbox_inches="tight")


def plot_trajectories_calibrated_model():
    # Pre-equilibration
    time_eq = np.linspace(0, 30, 30)
    pars_eq = np.copy(par_set_calibrated)
    pars_eq[[24, 25]] = 0  # Setting catalytic reactions to zero for pre-equilibration
    eq_conc1 = pre_equilibration(model, time_eq, pars_eq)[1]
    tspan = np.linspace(0, 60, 100)
    sim2 = ScipyOdeSimulator(model, tspan, param_values=par_set_calibrated, initials=eq_conc1).run().all

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
    axarr[5].plot(tspan, sim2['__s15'], color='#D55E00', label='MKK7:(Tyr)JNK3')
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

    plt.savefig('model_trajectories_calibrated.pdf', format='pdf', bbox_inches='tight')


def plot_arrestin_noarrestin_ppjnk3():
    # Pre-equilibration
    time_eq = np.linspace(0, 100, 100)
    pars_eq = np.copy(par_set_calibrated)
    pars_eq[[36, 37]] = 0  # Setting catalytic reactions to zero for pre-equilibration
    eq_conc1 = pre_equilibration(model, time_eq, pars_eq)[1]
    tspan = np.linspace(0, 60, 100)
    sim2 = ScipyOdeSimulator(model, tspan, param_values=par_set_calibrated, initials=eq_conc1).run().all

    # No arrestin experiments
    # Pre equilibration
    pars_eq[arrestin_idx] = 0
    pars_eq[jnk3_initial_idxs] = [0.492, 0.108, 0]
    eq_conc2 = pre_equilibration(model, time_eq, pars_eq)[1]

    par_set_calibrated[arrestin_idx] = 0
    par_set_calibrated[jnk3_initial_idxs] = [0.492, 0.108, 0]
    sim3 = ScipyOdeSimulator(model, tspan, param_values=par_set_calibrated, initials=eq_conc2).run().all

    rate_arr3 = np.diff(sim2['pTyr_jnk3']) / np.diff(tspan)
    rate_noarr3 = np.diff(sim3['pTyr_jnk3']) / np.diff(tspan)

    arr3_idx = np.argmax(rate_arr3)
    no_arr3_idx = np.argmax(rate_noarr3)
    # print (rate_arr3)
    # print (rate_noarr3)
    # print ('arr3', np.argmax(rate_arr3), tspan[arr3_idx])
    # print ('no_arr3', max(rate_noarr3), tspan[no_arr3_idx])
    print (sim2['pTyr_jnk3'][-1], sim3['pTyr_jnk3'][-1])

    plt.plot(tspan, sim2['all_jnk3'], color='r', label='ppJNK3 with Arrestin-3')
    plt.plot(tspan, sim3['all_jnk3'], color='k', label='ppJNK3 no Arrestin-3')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Concentration [$\mu$M]')
    plt.legend()
    plt.savefig('arrestin_noarrestin_ppjnk3_1.pdf', format='pdf', bbox_inches='tight')

def plot_uujnk3_production():
    ## This function requires a model observables of the single phosphorylated jnk3 only upjnk3, pujnk3
    # Pre-equilibration
    time_eq = np.linspace(0, 100, 100)
    pars_eq = np.copy(par_set_calibrated)
    pars_eq[[36, 37]] = 0  # Setting catalytic reactions to zero for pre-equilibration
    eq_conc1 = pre_equilibration(model, time_eq, pars_eq)[1]
    tspan = np.linspace(0, 60, 100)
    sim2 = ScipyOdeSimulator(model, tspan, param_values=par_set_calibrated, initials=eq_conc1).run().all

    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(8, 6))
    fig.subplots_adjust(hspace=0.5)
    axes[0].set_title('JNK3 activation')
    axes[0].plot(tspan, sim2['all_jnk3'], color="#999999", label='ppJNK3')
    axes[0].plot(tspan, sim2['pThr_jnk3'] - sim2['__s4'], color="#E69F00", label='pThr-JNK3:Arrestin-3')
    axes[0].plot(tspan, sim2['pTyr_jnk3'] - sim2['__s5'], color="#56B4E9", label='pTyr-JNK3:Arrestin-3')
    axes[0].set_ylabel(r'Concentration [$\mu$M]', fontsize=12)
    # plt.xlim(0, tspan[-1])
    axes[0].legend(loc=0, frameon=False)

    par_name_idx = {j.name: i for i, j in enumerate(model.parameters)}

    # Selecting the reactions in which doubly phosphorylated JNK3 is involved
    rxns_pjnk3 = OrderedDict()
    rxns_pjnk3['mkk4 first -Arr'] = model.reactions_bidirectional[21]['rate']
    rxns_pjnk3['mkk7 first -Arr'] = model.reactions_bidirectional[23]['rate']
    rxns_pjnk3['mkk4 first Arr'] = model.reactions_bidirectional[25]['rate']
    rxns_pjnk3['mkk7 first Arr'] = model.reactions_bidirectional[27]['rate']

    rxns_ppjnk3 = OrderedDict()
    rxns_ppjnk3['mkk4 sec -Arr'] = model.reactions_bidirectional[22]['rate']
    rxns_ppjnk3['mkk7 sec -Arr'] = model.reactions_bidirectional[24]['rate']
    rxns_ppjnk3['mkk4 sec Arr'] = model.reactions_bidirectional[26]['rate']
    rxns_ppjnk3['mkk7 sec Arr'] = model.reactions_bidirectional[28]['rate']

    colors = ["#009E73", "#0072B2", "#D55E00", "#CC79A7"]

    counter = 0
    for label, rr in rxns_pjnk3.items():
        mon = rr
        var_to_study = [atom for atom in mon.atoms(sympy.Symbol)]
        arg_f1 = [0] * len(var_to_study)
        for idx, va in enumerate(var_to_study):
            if str(va).startswith('__'):
                sp_idx = int(''.join(filter(str.isdigit, str(va))))
                arg_f1[idx] = np.maximum(0, sim2['__s{0}'.format(sp_idx)])
            else:
                arg_f1[idx] = par_set_calibrated[par_name_idx[va.name]]

        f1 = sympy.lambdify(var_to_study, mon)
        mon_values = f1(*arg_f1)
        # print (label, mon_values)
        axes[1].semilogy(tspan, mon_values, label=label, color=colors[counter])
        counter += 1
    # axes[1].legend(loc=0, ncol=2, frameon=False)
    axes[1].set_title('JNK3 first phosphorylation reactions')
    axes[1].set_ylabel(r'Rate [$\mu$M/s]', fontsize=12)
    # axes[1].set_xlabel('Time(s)', fontsize=14)
    # plt.xlim(0, tspan[-1])

    counter = 0
    for label, rr in rxns_ppjnk3.items():
        mon = rr
        var_to_study = [atom for atom in mon.atoms(sympy.Symbol)]
        arg_f1 = [0] * len(var_to_study)
        for idx, va in enumerate(var_to_study):
            if str(va).startswith('__'):
                sp_idx = int(''.join(filter(str.isdigit, str(va))))
                arg_f1[idx] = np.maximum(0, sim2['__s{0}'.format(sp_idx)])
            else:
                arg_f1[idx] = par_set_calibrated[par_name_idx[va.name]]

        f1 = sympy.lambdify(var_to_study, mon)
        mon_values = f1(*arg_f1)
        # print (label, mon_values)
        axes[2].semilogy(tspan, mon_values, label=label, color=colors[counter])
        counter += 1
    axes[2].set_title('JNK3 second phosphorylation reactions')
    axes[2].legend(loc=0, ncol=2, frameon=False)
    axes[2].set_ylabel(r'Rate [$\mu$M/s]', fontsize=12)
    axes[2].set_xlabel('Time(s)', fontsize=14)

    plt.savefig('ppjnk3_production_3.pdf', format='pdf', bbox_inches='tight')



# plot_trajectories_calibrated_model()
# plot_trajectories_nocalibrated_model()
plot_arrestin_noarrestin_ppjnk3()
# plot_uujnk3_production()