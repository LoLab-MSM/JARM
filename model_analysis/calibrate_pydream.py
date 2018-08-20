import numpy as np
from pydream.core import run_dream
from pysb.simulator import ScipyOdeSimulator
from pydream.parameters import SampledParam
from scipy.stats import norm, halfnorm, uniform
from pydream.convergence import Gelman_Rubin
from jnk3_no_ask1 import model
import pandas as pd
from equilibration_function import pre_equilibration

# Initialize PySB solver

exp_data = pd.read_csv('../data/exp_data_3min.csv')

tspan = np.linspace(0, exp_data['Time (secs)'].values[-1], 181)
t_exp_mask = [idx in exp_data['Time (secs)'].values[:] for idx in tspan]

solver = ScipyOdeSimulator(model, tspan=tspan)

like_mkk4_arrestin_pjnk3 = norm(loc=exp_data['pTyr_arrestin_avg'].values,
                                scale=exp_data['pTyr_arrestin_std'].values)

like_mkk7_arrestin_pjnk3_04 = halfnorm(loc=exp_data['pTyr_arrestin_avg'].values[:4],
				scale=exp_data['pTyr_arrestin_std'].values[:4])

like_mkk7_arrestin_pjnk3 = norm(loc=exp_data['pThr_arrestin_avg'].values[4:],
                                scale=exp_data['pThr_arrestin_std'].values[4:])

like_mkk4_noarrestin_pjnk3 = norm(loc=exp_data['pTyr_noarrestin_avg'].values,
                                scale=exp_data['pTyr_noarrestin_std'].values)
like_mkk7_noarrestin_pjnk3 = norm(loc=exp_data['pThr_noarrestin_avg'].values,
                                scale=exp_data['pThr_noarrestin_std'].values)



# Add PySB rate parameters to be sampled as unobserved random variables to DREAM with normal priors

## New kds in jnk3 mkk4/7
# idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream
# idx_pars_calibrate = [5, 9, 11, 15, 17, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream2
idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 19, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream3

rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

# Index of Initial conditions of Arrestin
arrestin_idx = [44]
jnk3_initial_value = 0.6  # total jnk3
jnk3_initial_idxs = [47, 48, 49]
kcat_idx = [36, 37]

param_values = np.array([p.value for p in model.parameters])

sampled_parameter_names = [SampledParam(norm, loc=np.log10(par), scale=2) for par in param_values[rates_of_interest_mask]]
# We calibrate the pMKK4 - Arrestin-3 reverse reaction rate. We have experimental data
# for this interaction and know that the k_r varies from 160 to 1068 (standard deviation)
sampled_parameter_names[0] = SampledParam(uniform, loc=np.log10(120), scale=np.log10(1200)-np.log10(120))
sampled_parameter_names[6] = SampledParam(uniform, loc=np.log10(28), scale=np.log10(280)-np.log10(28))

nchains = 5
niterations = 300000


def likelihood(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    pars1 = np.copy(param_values)
    pars2 = np.copy(param_values)
    # Pre-equilibration
    time_eq = np.linspace(0, 100, 100)
    pars_eq1 = np.copy(param_values)
    pars_eq2 = np.copy(param_values)

    pars_eq2[arrestin_idx] = 0
    pars_eq2[jnk3_initial_idxs] = [0.5958, 0, 0.0042]

    all_pars = np.stack((pars_eq1, pars_eq2))
    all_pars[:, kcat_idx] = 0  # Setting catalytic reactions to zero for pre-equilibration
    try:
        eq_conc = pre_equilibration(model, time_eq, all_pars)[1]
    except:
        logp_total = -np.inf
        return logp_total

    # Simulating models with initials from pre-equilibration and parameters for condition with/without arrestin
    pars2[arrestin_idx] = 0
    pars2[jnk3_initial_idxs] = [0.5958, 0, 0.0042]
    sim = solver.run(param_values=[pars1, pars2], initials=eq_conc).all
    logp_mkk4_arrestin = np.sum(like_mkk4_arrestin_pjnk3.logpdf(sim[0]['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value))
    logp_mkk7_arrestin = np.sum(like_mkk7_arrestin_pjnk3.logpdf(sim[0]['pThr_jnk3'][t_exp_mask][4:] / jnk3_initial_value))
    logp_mkk7_arrestin_04 = np.sum(like_mkk7_arrestin_pjnk3_04.logpdf(sim[0]['pThr_jnk3'][t_exp_mask][:4] / jnk3_initial_value))
    logp_mkk7_arrestin_total = logp_mkk7_arrestin + logp_mkk7_arrestin_04
    # No arrestin simulations/experiments

    logp_mkk4_noarrestin = np.sum(like_mkk4_noarrestin_pjnk3.logpdf(sim[1]['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value))
    logp_mkk7_noarrestin = np.sum(like_mkk7_noarrestin_pjnk3.logpdf(sim[1]['pThr_jnk3'][t_exp_mask] / jnk3_initial_value))

    #If model simulation failed due to integrator errors, return a log probability of -inf.
    logp_total = logp_mkk4_arrestin + logp_mkk7_arrestin + logp_mkk4_noarrestin + logp_mkk7_noarrestin
    if np.isnan(logp_total):
        logp_total = -np.inf

    return logp_total

par1 = np.load('calibrated_pars_pso.npy')
par2 = np.load('calibrated_pars_pso2.npy')
par3 = np.load('calibrated_pars_pso3.npy')
par4 = np.load('calibrated_pars_pso4.npy')
par5 = np.load('calibrated_pars_pso5.npy')
pso_pars = [par1, par2, par3, par4, par5]

if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = niterations
    sampled_params, log_ps = run_dream(parameters=sampled_parameter_names, likelihood=likelihood, start=pso_pars,
                                       niterations=niterations, nchains=nchains, multitry=False,
                                       gamma_levels=4, adapt_gamma=True, history_thin=1,
                                       model_name='jnk3_dreamzs_5chain2', verbose=False)

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
                                               adapt_gamma=True, history_thin=1, model_name='jnk3_dreamzs_5chain2',
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
                  'multitry':False, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':'jnk3_dreamzs_5chain2', 'verbose':False}
