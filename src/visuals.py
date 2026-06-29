# %% [1] Key Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mplsoccer import Pitch


# %% [2] Passing network visualizations

def _draw_network_base(graph, positions, player_weight_totals, title):
    """
    Drawing logic for the passing network: pitch setup, edges, nodes,
    names, and colorbar. Used by both plot_passing_network (intact network)
    and plot_passing_network_after_removal (network minus the playmaker).

    Returns (fig, ax, pitch)
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

    # Node size also scaled by weight_per_90
    min_size, max_size = 400, 1800
    if node_weights.max() > node_weights.min():
        size_norm = (node_weights - node_weights.min()) / (node_weights.max() - node_weights.min())
    else:
        size_norm = np.zeros_like(node_weights)
    node_sizes = min_size + size_norm * (max_size - min_size)

    node_xs = [positions[pid][0] for pid in node_ids]
    node_ys = [positions[pid][1] for pid in node_ids]

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
    Draws a passing network on a horizontal pitch (mplsoccer, attacking goal on the right).

    - Nodes are placed at each player's average pitch position.
    - Node color follows a gradient based on weight_per_90, so more active and dangerous
      players stand out visually.
    - Node size is also scaled by weight_per_90.
    - Edge thickness is proportional to edge weight.
    - Player names are shown in small text below each node.

    Parameters:
        graph: networkx.DiGraph, as returned by build_passing_network
        positions: dict {playerId: (x, y)}, as returned by get_average_player_positions
        player_weight_totals: DataFrame with columns 'playerId', 'shortName',
            'weight_per_90' (as returned by get_player_pass_weight_totals)
        title: plot title

    Returns the matplotlib (fig, ax) tuple.
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
    used to be.

    Parameters:
        graph_after: networkx.DiGraph, the network with the playmaker already
            removed (as returned by simulate_playmaker_removal)
        positions: dict {playerId: (x, y)} for the remaining players 
        player_weight_totals: same as in plot_passing_network
        playmaker_id: the removed player's id
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


 
# %% [3] Forward danger-zone receptions visualization
 
def plot_forward_receptions_in_danger_zone(
    receptions_by_forward, df_players, danger_zone_center, danger_zone_radius,
    title="Forward Receptions in the Dangerous Zone"
):
    """
    Draws two side-by-side pitches (before/after removing the playmaker),
    showing every pass reception a forward made inside the "dangerous zone"
    (a circle near the opponent's goal, see get_forward_receptions_in_danger_zone)
    as a dot, colored by forward, with a reception count shown in the legend.
 
    Parameters:
        receptions_by_forward: dict {forward_id: {'before': [(x,y),...], 'after': [(x,y),...]}},
            as returned by get_forward_receptions_in_danger_zone
        df_players: DataFrame with 'wyId' and 'shortName', for legend labels
        danger_zone_center: (x, y) tuple, the circle's center
        danger_zone_radius: float, the circle's radius
        title: overall figure title
 
    Returns the matplotlib (fig, axes) tuple.
    """
    pitch = Pitch(pitch_type='wyscout', pitch_color='grass', line_color='white', pad_right=20)
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(14, 7))
 
    name_lookup = df_players.set_index('wyId')['shortName'].to_dict()
    forward_ids = list(receptions_by_forward.keys())
    color_cycle = plt.colormaps['tab10']
    forward_colors = {fwd_id: color_cycle(i % 10) for i, fwd_id in enumerate(forward_ids)}
 
    cx, cy = danger_zone_center
 
    for ax, period in zip(axes, ['before', 'after']):
        # Draw the dangerous-zone circle as a dashed reference outline
        circle = plt.Circle(
            (cx, cy), danger_zone_radius, fill=False, edgecolor='yellow',
            linestyle='--', linewidth=1.5, zorder=1
        )
        ax.add_patch(circle)
 
        for fwd_id in forward_ids:
            points = receptions_by_forward[fwd_id][period]
            name = name_lookup.get(fwd_id, str(fwd_id))
            count = len(points)
            color = forward_colors[fwd_id]
 
            if points:
                xs, ys = zip(*points)
            else:
                xs, ys = [], []
 
            pitch.scatter(
                xs, ys, s=90, color=color, edgecolors='black', linewidth=0.8,
                alpha=0.85, zorder=2, ax=ax, label=f"{name} ({count})"
            )
 
        ax.set_title(period.capitalize(), fontsize=14, color='white', pad=8)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
 
    fig.suptitle(title, fontsize=16, color='white', y=0.98)
    fig.patch.set_facecolor('#1a1a1a')
 
    return fig, axes
 