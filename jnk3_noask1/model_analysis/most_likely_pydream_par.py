import numpy as np

chain0 = np.load('pydream_results/jnk3_dreamzs_5chain_sampled_params_chain_0_100000.npy')
chain1 = np.load('pydream_results/jnk3_dreamzs_5chain_sampled_params_chain_1_100000.npy')
chain2 = np.load('pydream_results/jnk3_dreamzs_5chain_sampled_params_chain_2_100000.npy')
chain3 = np.load('pydream_results/jnk3_dreamzs_5chain_sampled_params_chain_3_100000.npy')
chain4 = np.load('pydream_results/jnk3_dreamzs_5chain_sampled_params_chain_4_100000.npy')

total_iterations = chain0.shape[0]
burnin = total_iterations / 2
samples = np.concatenate((chain0[burnin:, :], chain1[burnin:, :], chain2[burnin:, :],
                          chain3[burnin:, :], chain4[burnin:, :]))

u, indices, counts = np.unique(samples, return_index=True, return_counts=True, axis=0)

max_idx = np.argmax(counts)

np.save('pydream_most_likely_100000.npy', u[max_idx])
