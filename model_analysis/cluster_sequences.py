from tropical.dynamic_signatures_range import run_tropical_multi
from tropical.clustering import ClusterSequences, PlotSequences
from jnk3_no_ask1 import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator

params = np.array([par.value for par in model.parameters])
tspan = np.linspace(0, 60, 100)
sim = ScipyOdeSimulator(model, tspan=tspan).run(param_values=[params, params, params])

signatures = run_tropical_multi(model, sim)
s27 = signatures[27]['consumption']

cs = ClusterSequences(s27)
cs.diss_matrix()
cs.agglomerative_clustering(2)
PS = PlotSequences(cs)
PS.plot_sequences(type_fig='entropy')
