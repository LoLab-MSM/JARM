import numpy as np
from pydream.core import run_dream
from pysb.simulator import ScipyOdeSimulator
from pydream.parameters import SampledParam
from scipy.stats import norm, uniform
from pydream.convergence import Gelman_Rubin
from jnk3_no_ask1_final import model
import pandas as pd
from equilibration_function import pre_equilibration

# Initialize PySB solver

exp_data = pd.read_csv('../../data/exp_data_arrestin_normalization_1h_138max.csv')

ignore = 1

tspan = np.linspace(0, exp_data['Time (secs)'].values[-(ignore+1)], 121)
t_exp_mask = [idx in exp_data['Time (secs)'].values[:-1] for idx in tspan]
solver = ScipyOdeSimulator(model, tspan=tspan)

like_mkk4_arrestin_pjnk3 = norm(loc=exp_data['pTyr_arrestin_avg'].values[:-ignore] + np.finfo(float).eps,
                                scale=exp_data['pTyr_arrestin_std'].values[:-ignore])
like_mkk7_arrestin_pjnk3 = norm(loc=exp_data['pThr_arrestin_avg'].values[:-ignore] + np.finfo(float).eps,
                                scale=exp_data['pThr_arrestin_std'].values[:-ignore])

like_mkk4_noarrestin_pjnk3 = norm(loc=exp_data['pTyr_noarrestin_avg'].values[:-ignore] + np.finfo(float).eps,
                                scale=exp_data['pTyr_noarrestin_std'].values[:-ignore])
like_mkk7_noarrestin_pjnk3 = norm(loc=exp_data['pThr_noarrestin_avg'].values[:-ignore] + np.finfo(float).eps,
                                scale=exp_data['pThr_noarrestin_std'].values[:-ignore])


# Add PySB rate parameters to be sampled as unobserved random variables to DREAM with normal priors

#New kds in jnk3 mkk4/7
idx_pars_calibrate = [1, 15, 17, 19, 24, 25, 26, 27]
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

arrestin_idx = [32]
jnk3_initial_value = 0.6
jnk3_initial_idxs = [35, 36, 37]

param_values = np.array([p.value for p in model.parameters])

sampled_parameter_names = [SampledParam(norm, loc=np.log10(par), scale=2) for par in param_values[rates_of_interest_mask]]
# We calibrate the pMKK4 - Arrestin-3 reverse reaction rate. We have experimental data
# for this interaction and know that the k_r varies from 160 to 1068 (standard deviation)
sampled_parameter_names[0] = SampledParam(uniform, loc=np.log10(160), scale=np.log10(1068)-np.log10(160))

nchains = 5
niterations = 50000


def likelihood(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    pars1 = np.copy(param_values)
    pars2 = np.copy(param_values)
    # Pre-equilibration
    time_eq = np.linspace(0, 30, 30)
    pars_eq1 = np.copy(param_values)
    pars_eq2 = np.copy(param_values)

    pars_eq2[arrestin_idx] = 0
    pars_eq2[jnk3_initial_idxs] = [0.492, 0.108, 0]

    all_pars = np.stack((pars_eq1, pars_eq2))
    all_pars[:, [24, 25]] = 0  # Setting catalytic reactions to zero for pre-equilibration
    eq_conc = pre_equilibration(model, time_eq, all_pars)[1]

    # Simulating models with initials from pre-equilibration and parameters for condition with/without arrestin
    pars2[arrestin_idx] = 0
    pars2[jnk3_initial_idxs] = [0.492, 0.108, 0]
    sim = solver.run(param_values=[pars1, pars2], initials=eq_conc).all
    logp_mkk4_arrestin = np.sum(like_mkk4_arrestin_pjnk3.logpdf(sim[0]['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value))
    logp_mkk7_arrestin = np.sum(like_mkk7_arrestin_pjnk3.logpdf(sim[0]['pThr_jnk3'][t_exp_mask] / jnk3_initial_value))

    # No arrestin simulations/experiments

    logp_mkk4_noarrestin = np.sum(like_mkk4_noarrestin_pjnk3.logpdf(sim[1]['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value))
    logp_mkk7_noarrestin = np.sum(like_mkk7_noarrestin_pjnk3.logpdf(sim[1]['pThr_jnk3'][t_exp_mask] / jnk3_initial_value))

    #If model simulation failed due to integrator errors, return a log probability of -inf.
    logp_total = logp_mkk4_arrestin + logp_mkk7_arrestin + logp_mkk4_noarrestin + logp_mkk7_noarrestin
    if np.isnan(logp_total):
        logp_total = -np.inf

    return logp_total

# We can start the chains from pso calibrated parameters to converge the chains faster
# pso_pars0 = np.load('jnk3_noASK1_calibrated_pars0.npy')
# pso_pars2 = np.load('jnk3_noASK1_calibrated_pars2.npy')
# pso_pars3 = np.load('jnk3_noASK1_calibrated_pars3.npy')
# pso_pars5 = np.load('jnk3_noASK1_calibrated_pars5.npy')
# pso_pars6 = np.load('jnk3_noASK1_calibrated_pars6.npy')
#
# pso_pars = [pso_pars0, pso_pars2, pso_pars3, pso_pars5, pso_pars6]

if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = niterations
    sampled_params, log_ps = run_dream(parameters=sampled_parameter_names, likelihood=likelihood,
                                       niterations=niterations, nchains=nchains, multitry=False,
                                       gamma_levels=4, adapt_gamma=True, history_thin=1,
                                       model_name='jnk3_dreamzs_5chain', verbose=True)

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save('pydream_results/jnk3_dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
        np.save('pydream_results/jnk3_dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

    #Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ',total_iterations,' GR = ',GR)
    np.savetxt('pydream_results/jnk3_dreamzs_5chain_GelmanRubin_iteration_'+str(total_iterations)+'.txt', GR)

    old_samples = sampled_params
    if np.any(GR>1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations
            sampled_params, log_ps = run_dream(parameters=sampled_parameter_names, likelihood=likelihood,
                                               niterations=niterations, nchains=nchains, start=starts, multitry=False, gamma_levels=4,
                                               adapt_gamma=True, history_thin=1, model_name='jnk3_dreamzs_5chain',
                                               verbose=True, restart=True)


            # Save sampling output (sampled parameter values and their corresponding logps).
            for chain in range(len(sampled_params)):
                np.save('pydream_results/jnk3_dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
                np.save('pydream_results/jnk3_dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ',total_iterations,' GR = ',GR)
            np.savetxt('pydream_results/jnk3_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations)+'.txt', GR)

            if np.all(GR<1.2):
                converged = True

    try:
        #Plot output
        import seaborn as sns
        from matplotlib import pyplot as plt
        total_iterations = len(old_samples[0])
        burnin = total_iterations/2
        samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :],
                                  old_samples[3][burnin:, :], old_samples[4][burnin:, :]))

        ndims = len(sampled_parameter_names)
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            fig = plt.figure()
            sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True)
            fig.savefig('pydream_results/PyDREAM_jnk3_dimension_'+str(dim))

    except ImportError:
        pass

else:

    run_kwargs = {'parameters':sampled_parameter_names, 'likelihood':likelihood, 'niterations':niterations, 'nchains':nchains, \
                  'multitry':False, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':'jnk3_dreamzs_5chain', 'verbose':False}
