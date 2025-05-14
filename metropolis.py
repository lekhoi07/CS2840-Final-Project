import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import PchipInterpolator

class MetropolisSimulation:
    def __init__(self, data, input_genes, timepoints, clusters, temperature: float = 300.0,
                 max_steps: int = 10000, move_size: float = 0.1, euler_step_size: float = 0.1,
                 decay_rate: float = 0.1, output_prefix: str = None, save_frequency: int = 1, interpolation: bool = True):
        """
        Initialize Metropolis simulation object.

        Args:
            data: anndata object that is scRNA-seq data
            input_genes: genes whose relationships we are trying to infer
            timepoints: list of stages whose expression to consider
            clusters: list of clusters whose expression to consider
            temperature: Temperature in Kelvin
            max_steps: Maximum number of simulation steps
            move_size: Variance of random moves in proposal function
            euler_step_size: step size for euler approximation when calculating energy
            decay_rate: rate of decay for mRNA
            output_prefix: Prefix for output files
            save_frequency: How often to save structures (in terms of accepted 
            moves)
            interpolation: whether to interpolate values between the discrete timepoints in dataset
        """
        # sim settings
        self.temperature = temperature
        self.max_steps = max_steps
        self.kT = 0.00198720425864083 * temperature
        self.move_size = move_size
        self.accepted_moves = 0
        self.total_moves = 0
        self.step_size = euler_step_size
        self.decay_rate = decay_rate

        # settings
        self.output_prefix = output_prefix
        self.input_genes = input_genes
        self.save_frequency = save_frequency

        # mapping of timepoint labels to numerical time
        self.timepoints = timepoints
        timepoint_dict = {
            "Sp1": 2.75,
            "Sp2": 6.25,
            "Sp3": 7.25,
            "EB": 11.45,
            "HB": 13.5, 
            "MB": 19,
            "EG": 24,
            "LG": 27
        }
        self.numerical_timepoints = [timepoint_dict[timepoint] for timepoint in timepoints]
        self.start_time = self.numerical_timepoints[0]

        # initialize current state
        self.initial_expressions, self.actual_expressions, self.max_expressions = self.get_expressions(input_genes, timepoints, clusters, data)
        self.interpolation = interpolation
        if interpolation:
            self.interpolated_expressions = self.get_interpolated_expressions(self.actual_expressions, self.numerical_timepoints, self.step_size)
        self.current_weights = None
        self.current_energy = None


    # Calculate initial, actual, and max expression values for each gene across timepoints
    def get_expressions(self, genes, timepoints, clusters, data):
        initial_expressions = np.zeros(len(genes))
        actual_expressions = np.zeros((len(genes), len(timepoints)))
        max_expressions = []
        for i in range(len(genes)):
            gene = genes[i]
            gene_ind = data.var_names.get_loc(gene)
            cells_of_interest = data.obs[data.obs['seurat_clusters'].isin(clusters)]
            avg_expressions = []
            expressions = []
            for j, timepoint in enumerate(timepoints):
                timepoint_cells_ind = cells_of_interest[cells_of_interest['orig.ident'] == timepoint].index
                expression = data.raw[timepoint_cells_ind, gene_ind].X.toarray().flatten()
                avg_expressions.append(np.median(expression))
                expressions.append(expression)
                if j == 0:
                    initial_expressions[i] = np.median(expression)
            
            actual_expressions[i] = avg_expressions
            max_expressions.append(np.max(np.concatenate(expressions)))
        return initial_expressions, actual_expressions, np.array(max_expressions)


    # Helper function to do interpolation of expression values between discrete timepoints
    def get_interpolated_expressions(self, actual_expressions, numerical_timepoints, euler_step_size):
        start_time = self.start_time
        num_steps = int(np.ceil((max(numerical_timepoints) - start_time) / euler_step_size)) + 1
        euler_times = np.linspace(start_time, start_time + num_steps * euler_step_size, num_steps)
        interpolated_expressions = np.zeros((len(self.input_genes), len(euler_times)))
        for i in range(len(self.input_genes)):
            interpolator = PchipInterpolator(numerical_timepoints, actual_expressions[i])
            interpolated_expressions[i] = np.clip(interpolator(euler_times), 0, None)
        return interpolated_expressions


    def _save_trajectory(self, filename: str, trajectory: List[Tuple[np.ndarray, float]]):
        """
        Save trajectory to file.

        Args:
            filename: Output filename
            trajectory: List of (weights, energy) tuples
        """
        weights, energy = trajectory[-1]
        df = pd.DataFrame(weights, index=self.input_genes, columns=self.input_genes)
        df.to_csv(filename)
        return energy


    def propose_move(self, weights: np.ndarray, bias=0) -> np.ndarray:
        """
        Propose a new state by making a random move from the current state.
        Randomly selects an edge weight and moves it in a random direction.

        Args:
            weights: Current weights (N x N numpy array)
            bias: Optional bias for the random move (default: 0)

        Returns:
            New proposed weights (N x N numpy array)
        """
        weights_copy = weights.copy()
        i = np.random.randint(len(self.input_genes))
        j = np.random.randint(len(self.input_genes))
        offset = np.random.normal(bias, self.move_size)
        weights_copy[i , j] = weights_copy[i , j] + offset
        return weights_copy
        
    
    # sigmoid function with range from (0, 1)
    def sigmoid_helper(self, x):
        return 0.5 * (x / np.sqrt(x ** 2 + 1) + 1)


    def calculate_energy(self, genes, initial_expressions, numerical_timepoints, weights: np.ndarray, max_expressions, step_size, actual_expressions, decay_rate) -> float:
        """
        Calculate energy using actual expression values versus predicted from the paramaters via Euler's method

        Args:
            genes: genes of interest
            initial_expressions: initial condition of Euler's method
            numerical_timepoints: hours at which we have actual expression values
            weights: weights array (N x N numpy array)
            max_expressions: maximum expression for each gene
            step_size: Euler's method step size
            actual_expressions: real expression values to calculate MSE against
            decay_rate: Parameter for decay of RNA

        Returns:
            Total energy of the current weights matrix (MSE of simulated and actual expressions)
        """
        sample_ind = 0

        # Euler's method
        start_time = self.start_time
        num_steps = int(np.ceil((max(numerical_timepoints) - start_time) / step_size)) + 1
        euler_times = np.linspace(start_time, start_time + num_steps * step_size, num_steps)
        if self.interpolation:
            predicted_expressions = np.zeros((len(genes), len(euler_times)))
        else:
            predicted_expressions = np.zeros((len(genes), len(numerical_timepoints)))
        current_expressions = initial_expressions.copy()
        for step in range(1, num_steps):
            total_reg_effect = weights @ current_expressions
            delta_expressions = max_expressions * self.sigmoid_helper(total_reg_effect) - decay_rate * current_expressions
            current_expressions = current_expressions + delta_expressions * step_size

            # extract actual expression values for later
            if self.interpolation:
                predicted_expressions[:, step] = current_expressions
            else:
                while sample_ind < len(numerical_timepoints) and euler_times[step] >= numerical_timepoints[sample_ind]:
                    predicted_expressions[:, sample_ind] = current_expressions
                    sample_ind += 1

                if sample_ind >= len(numerical_timepoints):
                    break

        # MSE of predicted versus actual expressions
        mse = np.mean((predicted_expressions - actual_expressions) ** 2)
        return mse
    

    def metropolis_step(self, bias=0) -> Tuple[np.ndarray, float, bool]:
        """
        Perform one step of the Metropolis algorithm.

        Args:
            bias: Bias to use for the proposal distribution. Default is 0.

        Returns:
            Tuple containing:
            - New weights
            - New energy
            - Whether the move was accepted
        """
        # increment total moves
        self.total_moves += 1

        # propose a new state and calculate its energy
        new_state = self.propose_move(self.current_weights, bias)
        new_energy = self.calculate_energy(self.input_genes, self.initial_expressions, self.numerical_timepoints, new_state, self.max_expressions, self.step_size, self.interpolated_expressions if self.interpolation else self.actual_expressions, self.decay_rate)

        # if poposal_corr is true, implement the corrected 
        # Metropolis-Hastings acceptance probability; otherwise, implement the 
        # uncorrected acceptance probability
        p_accept = np.exp((self.current_energy - new_energy) / self.kT)
        p_accept *= (np.exp((((-(np.linalg.norm(new_state - self.current_weights - bias) ** 2)) + np.linalg.norm(self.current_weights - new_state - bias) ** 2) / (2 * (self.move_size ** 2))))) ** -1
        p_accept = min(1, p_accept)

        # sample a boolean according to acceptance probability. If it is 
        # accepted, then update current_coordinates, current_energy, and 
        # increment accepted_moves. Additionally, include the following code to 
        # save the PDB file if output is configured and it's a saving 
        # iteration. Return self.current_coordinates, self.current_energy, and 
        # the acceptance boolean as a 3-tuple.
        accepted = np.random.random() < p_accept
        if accepted:
            self.current_weights = new_state
            self.current_energy = new_energy
            self.accepted_moves += 1

        if (self.output_prefix and self.accepted_moves % self.save_frequency == 0):
                output_file = f"{self.output_prefix}_{self.accepted_moves//self.save_frequency:04d}.txt"
                self._save_trajectory(
                    output_file, [(self.current_weights, self.current_energy)])
        
        return (self.current_weights, self.current_energy, accepted)


    def run_simulation(self, proposal_bias=0) -> List[Tuple[np.ndarray, float]]:
        """
        Run the complete Metropolis simulation.

        Args:
            proposal_bias: Bias to use for the proposal distribution. Default 
            is 0.

        Returns:
            List of (coordinates, energy) tuples for accepted states
        """
        # Copy the initial coordinates and calculate the initial energy. 
        # Start a running trajectory list with this initial (weight, 
        # energy) pair. Then, use the following code to save the first 
        # trajectory:
        self.current_weights = np.random.uniform(-1.0, 1.0, size=(len(self.input_genes), len(self.input_genes)))

        # np.fill_diagonal(self.current_weights, 0)

        trajectory_list = []
        self.current_energy = self.calculate_energy(self.input_genes, self.initial_expressions, self.numerical_timepoints, self.current_weights, self.max_expressions, self.step_size, self.interpolated_expressions if self.interpolation else self.actual_expressions, self.decay_rate)
        trajectory_list.append((self.current_weights, self.current_energy))

        if self.output_prefix:
            output_file = f"{self.output_prefix}_0000.txt"
            self._save_trajectory(
                output_file, [(self.current_weights, self.current_energy)])

        # run the simulation for max_steps using metropolis_step, 
        # updating the trajectory for each accepted move. Feel free to output 
        # any extra structures you want for analysis.
        for i in range(self.max_steps):
            step = self.metropolis_step(proposal_bias)
            if step[2]:
                trajectory_list.append((self.current_weights, self.current_energy))
                if i % 50 == 0:
                    print(f"\rStep {i}/{self.max_steps} | Current energy: {self.current_energy:.3f}", end="", flush=True)
        return trajectory_list


    def analyze_results(self, trajectory) -> dict:
        """
        Args:
            trajectory: List of (weights, energy) tuples

        Returns:
            Dictionary containing analysis results
        """
        energies = np.array([e for _, e in trajectory])

        # find minimum energy structure
        min_energy_idx = np.argmin(energies)
        min_energy_weights = trajectory[min_energy_idx][0]
        min_energy_value = trajectory[min_energy_idx][1]

        # save minimum energy structure if output is configured
        if self.output_prefix:
            output_file = f"{self.output_prefix}_min_energy.txt"
            self._save_trajectory(
                output_file, [(min_energy_weights, min_energy_value)])

        results = {
            "mean_energy": np.mean(energies),
            "std_energy": np.std(energies),
            "min_energy": np.min(energies),
            "max_energy": np.max(energies),
            "acceptance_rate": self.accepted_moves / self.total_moves,
            "min_energy_structure": min_energy_weights,
            "min_energy_value": min_energy_value
        }

        # plot accepted moves versus energy over course of simulation
        plt.plot(range(1, len(energies) + 1), energies)
        plt.xlabel("# accepted moves")
        plt.ylabel("energy")
        plt.savefig(f"{self.output_prefix}_plot.png")
        plt.close()
        return results

    
