# %% [1] Key Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mplsoccer import Pitch


# %% [2] Passing network visualizations

def _draw_network_base(graph, positions, player_weight_totals, title):
    """
    Shared drawing logic for the passing network: pitch setup, edges, nodes,
    names, and colorbar. Used by both plot_passing_network (intact network)
    and plot_passing_network_after_removal (network minus the playmaker).

    Returns (fig, ax, pitch) so callers can add further elements (like the
    ghost node for the removed playmaker) on top before returning.
    """
    pitch = Pitch(pitch_type='wyscout', pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(12, 8))

    # Lookup for weight_per_90 and player names, restricted to players actually in the graph
    weight_lookup = player_weight_totals.set_index('playerId')['weight_per_90'].to_dict()
    name_lookup = player_weight_totals.set_index('playerId')['shortName'].to_dict()

    node_ids = list(graph.nodes())
    node_weights = np.array([weight_lookup.get(pid, 0.0) for pid in node_ids])

    # Color gradient based on weight_per_90
    norm = Normalize(vmin=node_weights.min(), vmax=node_weights.max())
    cmap = plt.colormaps['YlOrRd']
    node_colors = [cmap(norm(w)) for w in node_weights]

    # Node size also scaled by weight_per_90 (min/max bounds keep small-weight
    # nodes visible while still emphasizing the higher-weight ones)
    min_size, max_size = 400, 1800
    if node_weights.max() > node_weights.min():
        size_norm = (node_weights - node_weights.min()) / (node_weights.max() - node_weights.min())
    else:
        size_norm = np.zeros_like(node_weights)
    node_sizes = min_size + size_norm * (max_size - min_size)

    node_xs = [positions[pid][0] for pid in node_ids]
    node_ys = [positions[pid][1] for pid in node_ids]

    # Draw edges first (so nodes are layered on top), thickness proportional to weight
    edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    min_lw, max_lw = 0.5, 8

    for u, v in graph.edges():
        weight = graph[u][v]['weight']
        lw = min_lw + (weight / max_edge_weight) * (max_lw - min_lw)
        pitch.lines(
            positions[u][0], positions[u][1],
            positions[v][0], positions[v][1],
            lw=lw, color='white', alpha=0.6, zorder=1,
            ax=ax, comet=False
        )

    # Draw nodes
    pitch.scatter(
        node_xs, node_ys, s=node_sizes, c=node_colors,
        edgecolors='black', linewidth=1.5, zorder=2, ax=ax
    )

    # Player names below each node
    for pid, x, y in zip(node_ids, node_xs, node_ys):
        name = name_lookup.get(pid, str(pid))
        ax.annotate(
            name, xy=(x, y), xytext=(0, -12), textcoords='offset points',
            ha='center', va='top', fontsize=8, color='white', weight='bold', zorder=3
        )

    # Colorbar legend for the weight_per_90 gradient
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Weight per 90 minutes', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')

    ax.set_title(title, fontsize=16, color='white', pad=10)
    fig.patch.set_facecolor('#1a1a1a')

    return fig, ax, pitch


def plot_passing_network(graph, positions, player_weight_totals, title="Passing Network"):
    """
    Draws a passing network on a horizontal pitch (mplsoccer, broadcast-style
    orientation, attacking goal on the right).

    - Nodes are placed at each player's average pitch position.
    - Node color follows a gradient based on weight_per_90 (from
      player_weight_totals), so more "dangerous" playmakers stand out visually.
    - Node size is also scaled by weight_per_90, reinforcing the same signal.
    - Edge thickness is proportional to edge weight (sum of pass_weight
      between that specific pair of players).
    - Player names are shown in small text below each node.

    Parameters:
        graph: networkx.DiGraph, as returned by build_passing_network
        positions: dict {playerId: (x, y)}, as returned by get_average_player_positions
        player_weight_totals: DataFrame with columns 'playerId', 'shortName',
            'weight_per_90' (as returned by get_player_pass_weight_totals)
        title: plot title

    Returns the matplotlib (fig, ax) tuple, in case further customization
    or saving is needed.
    """
    fig, ax, _ = _draw_network_base(graph, positions, player_weight_totals, title)
    return fig, ax


def plot_passing_network_after_removal(
    graph_after, positions, player_weight_totals, playmaker_id, playmaker_position,
    title="Passing Network (After Man-Marking the Playmaker)"
):
    """
    Draws the passing network AFTER the playmaker has been removed (see
    simulate_playmaker_removal), using the same visual logic as
    plot_passing_network, plus a "ghost node" marking where the playmaker
    used to be: a dashed, hollow circle with his name, so a coach can see
    at a glance who was taken out and how the remaining network reacts
    around that gap.

    Parameters:
        graph_after: networkx.DiGraph, the network with the playmaker already
            removed (as returned by simulate_playmaker_removal)
        positions: dict {playerId: (x, y)} for the REMAINING players (the
            playmaker's own position is passed separately, see playmaker_position)
        player_weight_totals: same as in plot_passing_network
        playmaker_id: the removed player's id (used to look up his name for the label)
        playmaker_position: (x, y) tuple, the playmaker's average position
            before removal, used to place the ghost node
        title: plot title

    Returns the matplotlib (fig, ax) tuple.
    """
    fig, ax, pitch = _draw_network_base(graph_after, positions, player_weight_totals, title)

    name_lookup = player_weight_totals.set_index('playerId')['shortName'].to_dict()
    playmaker_name = name_lookup.get(playmaker_id, str(playmaker_id))

    ghost_x, ghost_y = playmaker_position
    pitch.scatter(
        [ghost_x], [ghost_y], s=1200, facecolors='none',
        edgecolors='red', linewidth=2.5, linestyle='--', zorder=2, ax=ax
    )
    ax.annotate(
        f"{playmaker_name}\n(removed)", xy=(ghost_x, ghost_y), xytext=(0, -16),
        textcoords='offset points', ha='center', va='top', fontsize=8,
        color='red', weight='bold', style='italic', zorder=3
    )

    return fig, ax