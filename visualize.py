import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import anndata


# make a box plot of gene expression for a given gene in the given clusters across all timepoints. Connect the medians with a line
def plot_gene_expression(data, gene, clusters):
        # get expression data
        gene_ind = data.var_names.get_loc(gene)
        cells_of_interest = data.obs[data.obs['seurat_clusters'].isin(clusters)]
        avg_expressions = []
        expression_data = []
        plt.figure(figsize=(10, 10))
        for timepoint in ["Sp1", "Sp2", "Sp3", "EB", "HB", "MB", "EG", "LG"]:
            timepoint_cells_ind = cells_of_interest[cells_of_interest['orig.ident'] == timepoint].index
            expression = data[timepoint_cells_ind, gene_ind].X.toarray().flatten()
            avg_expressions.append({'timepoint' : timepoint, 'expression' : np.median(expression)})
            for val in expression:
                expression_data.append({'timepoint' : timepoint, 'expression' : val})
        
        # Plot graph
        df = pd.DataFrame(expression_data)
        mean_df = pd.DataFrame(avg_expressions)
        seaborn.boxplot(df, x='timepoint', y='expression', palette='Set2')
        seaborn.lineplot(mean_df, x='timepoint', y='expression', markers='o', label='Median Expression')
        plt.title(f"{gene} Expression in Cluster {clusters}", fontsize=30)
        plt.ylabel('Normalized Expression', fontsize=30)
        plt.xlabel('Timepoint', fontsize=30)
        plt.legend(fontsize=15)
        plt.tight_layout()
        if gene == "Tcf/Lef":
            plt.savefig("Tcf-Lef_expression.png")
        else:
            plt.savefig(f"{gene}_expression.png")
        plt.close()


# makes plots of actual expression versus prediction expression 
def plot_simulated_expressions(numerical_timepoints, step_size, genes, initial_expressions, weights_csv, max_expressions, decay_rate, outfile, actual_expressions=None, interpolation = True):
    # Euler's method to find predicted expression
    weights = pd.read_csv(weights_csv, index_col=0)
    start_time = numerical_timepoints[0]
    num_steps = int(np.ceil((max(numerical_timepoints) - start_time) / step_size)) + 1
    euler_times = np.linspace(start_time, start_time + num_steps * step_size, num_steps)
    expression_trajectory = np.zeros((len(genes), num_steps))
    current_expressions = initial_expressions.copy()
    expression_trajectory[:, 0] = current_expressions
    for step in range(1, num_steps):
        total_reg_effect = weights @ current_expressions
        sigmoid_effect = 0.5 * (total_reg_effect / np.sqrt(total_reg_effect ** 2 + 1) + 1)
        delta_expressions = max_expressions * sigmoid_effect - decay_rate * current_expressions
        current_expressions = current_expressions + delta_expressions * step_size
        expression_trajectory[:, step] =  current_expressions

    # Make figure
    plt.figure(figsize=(10, 10))
    colors = plt.get_cmap('tab10')
    gene_colors = {gene : colors(i % 10) for i, gene in enumerate(genes)}
    for i, gene in enumerate(genes):
         color = gene_colors[gene]

         plt.plot(euler_times, np.log1p(expression_trajectory[i]), label=gene, color=color, linestyle='--')
         if actual_expressions is not None:
            if interpolation:
                plt.plot(euler_times, np.log1p(actual_expressions[i]), label=f"{gene} (actual)", color=color, linestyle='-')
            else:
                plt.scatter(numerical_timepoints, np.log1p(actual_expressions[i]), label=f"{gene} (actual)", marker='x', color=color)
    plt.ylabel("Expression (log1p-transformed)")
    plt.xlabel("Hours Past Fertilization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)  
    plt.close()  
    

# Make directed graph based on weights matrix
def plot_network(weights_csv, genes, outfile):
    weights = pd.read_csv(weights_csv, index_col=0)
    graph = nx.DiGraph()
    for gene in genes:
         graph.add_node(gene)
    for i in range(len(genes)):
         for j in range(len(genes)):
              graph.add_edge(genes[j], genes[i], weight=weights.iloc[i, j])
    
    # make different graphs that display only the top x% of genes for varying x
    for percentile in [0, 25, 50, 75]:
        nx.draw_networkx_nodes(graph, nx.circular_layout(graph))
        nx.draw_networkx_labels(graph, nx.circular_layout(graph))
        for x, y, z in graph.edges(data=True):
            nx.draw_networkx_edges(graph, nx.circular_layout(graph), edgelist=[(x, y)], 
                                    edge_color = 'blue' if float(z['weight']) > 0 else 'red', 
                                    alpha=abs(float(z['weight']) / np.max(np.abs(weights))) if abs(float(z['weight'])) > np.percentile(np.abs(weights), percentile) else 0, 
                                    connectionstyle=f'arc3,rad={0.15}', arrows=True)
        plt.tight_layout()
        plt.savefig(f"{outfile}_{percentile}.png")
        plt.close()

# Plot gene expression trajectories individually
print("loading data...")
data = anndata.read_h5ad("sp_data.h5ad")
plot_gene_expression(data, "Tcf/Lef", [16, 19])
plot_gene_expression(data, "PMAR1", [16, 19])
plot_gene_expression(data, "LOC592057", [16, 19])
plot_gene_expression(data, "Alx1", [16, 19])
plot_gene_expression(data, "SM50", [16, 19])