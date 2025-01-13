import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from seqanalysis.util.logging import config_logging
from IPython import embed

log = config_logging()


plt.rcParams["svg.fonttype"] = (
    "none"  # this is so that svg figures save text as text and not the single letters
)
plt.rcParams['font.size'] = 18

def plot_transition_matrix(matrix, labelx, labely, save_path, title):
    """
    Plot a heatmap of a transition matrix.

    Parameters:
    - matrix (array-like): The transition matrix to be visualized.
    - labels (list): Labels for the x and y axes.
    - save_path (str): File path to save the generated plot.
    - title (str): Title of the plot.
    """
    fig, ax = plt.subplots()
    hm = sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        vmin=0,
        vmax=100,
        fmt="d",
        cmap="Greys",
        xticklabels=labelx,
        yticklabels=labely,
    )
    ax.set_yticklabels(hm.get_yticklabels(), rotation=0)
    ax.set_xticklabels(hm.get_xticklabels(), rotation=45)
    ax.tick_params(left=False, bottom=False)
    sns.despine(top=False, right=False, left=False, bottom=False)
    ax.set_title(title)
    fig.tight_layout()
    log.info(f"Saving plot to {save_path}")
    fig.savefig(save_path, dpi=300)


def plot_transition_diagram(matrix, labels, node_size, edge_width, save_path, title):
    """
    Plot a transition diagram based on the given matrix and labels.

    Parameters:
    - matrix (array-like): The transition matrix to be visualized.
    - labels (list): Labels for the nodes in the diagram.
    - node_size (float): Size of the nodes in the diagram.
    - edge_width (float): Width scaling factor for the edges in the diagram.
    - save_path (str): File path to save the generated plot.
    - title (str): Title of the plot.
    """

    # Create a directed graph from the given matrix
    Graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

    # Map node labels to corresponding nodes
    node_labels = dict(zip(Graph, labels))

    # Get edge labels from the graph
    edge_labels = nx.get_edge_attributes(Graph, "weight")

    # Set the positions of nodes in a circular layout
    positions = nx.circular_layout(Graph)
    # mulitply by 10 to make the plot bigger

    # Create a subplot with a specified size and margins
    fig, ax = plt.subplots(figsize=(21 / 2.54, 19 / 2.54))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

    # Draw nodes with specified size, color, and transparency
    nx.draw_networkx_nodes(
        Graph,
        pos=positions,
        node_size=node_size,
        node_color="a7d7dd",
        ax=ax,
        alpha=0.9,
    )

    # Draw node labels
    nx.draw_networkx_labels(Graph, pos=positions, labels=node_labels)

    # Draw edges with specified width, arrows, and style
    edge_width = [x / edge_width for x in [*edge_labels.values()]]
    nx.draw_networkx_edges(
        Graph,
        pos=positions,
        node_size=node_size,
        width=edge_width,
        arrows=True,
        arrowsize=20,
        min_target_margin=25,
        min_source_margin=10,
        connectionstyle="arc3,rad=0.2",
        ax=ax,
    )

    # Draw edge labels at the midpoint of the edges
    nx.draw_networkx_edge_labels(
        Graph, positions, label_pos=0.5, edge_labels=edge_labels, ax=ax, rotate=False
    )

    # Remove spines for a cleaner appearance
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set the title of the plot
    plt.title(title)

    # Save the plot to the specified file path with a specified DPI
    log.info(f"Saving plot to {save_path}")
    fig.savefig(save_path, dpi=300)
