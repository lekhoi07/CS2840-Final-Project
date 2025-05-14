import anndata
from metropolis import MetropolisSimulation
from visualize import plot_network, plot_simulated_expressions

# read data
print("reading data...")
data = anndata.read_h5ad("sp_data.h5ad")
print("data read")

# initialization simulation
sim = MetropolisSimulation(data, input_genes=["Tcf/Lef", "PMAR1", "LOC592057", "Alx1", "SM50"], timepoints=["Sp1", "Sp2", "Sp3", "EB", "HB", "MB", "EG", "LG"],
                           clusters=[16,19], output_prefix="outs/", move_size=1.0, save_frequency=5000, max_steps=100000, decay_rate=1.0, interpolation = False)
print("starting simulation...")

# run simulation
trajectory = sim.run_simulation(proposal_bias=0.1)
print("\nsimulation finished")

# plot simulation results
results = sim.analyze_results(trajectory)
plot_network("outs/_min_energy.txt", sim.input_genes, "outs/network")
plot_simulated_expressions(sim.numerical_timepoints, sim.step_size, sim.input_genes, sim.initial_expressions, "outs/_min_energy.txt", sim.max_expressions, sim.decay_rate, "outs/simulated_expression.png",sim.interpolated_expressions if sim.interpolation else sim.actual_expressions, interpolation=sim.interpolation)
print("results plotted")
