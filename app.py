# %% [markdown]
# # Main Pipeline - Sports Data Science Final Project

# Quantification of the Tactical impact of man-marking and isolating playmakers through a Network Resilience Analysis. \
# _Livio Guerra (s4444159) - Leiden University._

# %% [1] Load Key Libraries and Custom Modules
# standard libraries
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import ipywidgets as widgets
from IPython.display import display

# data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch, VerticalPitch

# custom modules
import src.functions as fun
import src.visuals as viz

print("All libraries imported successfully!")

# %% [2] Load teams data

PROJECT_ROOT = os.getcwd()
df_teams = fun.load_teams_data(PROJECT_ROOT)

team_state = fun.run_team_selector(PROJECT_ROOT, df_teams)

# %% [3] Compute pass weights, minutes played, and player totals (per 90 minutes)
 
df_team_passes = team_state['df_team_passes']
selected_team_row = team_state['selected_team_row']
 
df_team_passes_weighted = fun.add_pass_weights(df_team_passes, alpha=0.6, beta=0.4, exponent=1.5)
 
df_players = fun.load_players_data(PROJECT_ROOT)
df_minutes = fun.get_season_minutes_played(PROJECT_ROOT, selected_team_row['wyId'], selected_team_row)
 
player_weight_totals = fun.get_player_pass_weight_totals(df_team_passes_weighted, df_players, df_minutes=df_minutes)

# %% [4] Select playmaker and build the lineup around him
 
lineup_state = fun.run_playmaker_lineup_builder(PROJECT_ROOT, selected_team_row, df_players, player_weight_totals)

# %% [5] Network resilience analysis 
# This cell automatically creates an HTML report, upon first run you will be provided with a comments empty file in the 'report\' folder.
# After inspecting the final analysis output you can add personal interpretations of outputs by editing the 'comments' section of this cell
# Re-run this cell to generate the final version of the report with the newly added comments.

playmaker_id = lineup_state['playmaker_id']
network_graph = lineup_state['network_graph']
player_positions = lineup_state['player_positions']
lineup_player_ids = {p['playerId'] for p in lineup_state['lineup']}
match_ids = [m['wyId'] for m in lineup_state['starting_matches']]

# Save the playmaker's position before removal (needed for the ghost node in the "after" plot)
playmaker_position = player_positions[playmaker_id]

network_graph_after = fun.simulate_playmaker_removal(network_graph, playmaker_id)
positions_after = {pid: pos for pid, pos in player_positions.items() if pid != playmaker_id}

# --- Collective passing output decay (replaces Global Efficiency, which is
#     near-zero on small, densely-connected 11-player networks) ---
lineup_weight_per_90 = fun.compute_lineup_weight_per_90(
    df_team_passes_weighted, lineup_player_ids, match_ids, PROJECT_ROOT,
    selected_team_row['wyId'], selected_team_row
)
weight_decay = fun.compute_network_weight_decay(lineup_weight_per_90, playmaker_id)
print(f"[INFO] Collective output: {weight_decay['weight_before']:.1f} -> "
      f"{weight_decay['weight_after']:.1f} ({weight_decay['percent_decay']:.1f}% decay)")

# --- Top weighted passing lanes (per 90 shared minutes), before and after ---
shared_minutes = fun.get_shared_minutes_by_pair(
    PROJECT_ROOT, selected_team_row['wyId'], selected_team_row, match_ids
)
top_edges_before = fun.get_top_weighted_edges(network_graph, df_players, shared_minutes, top_n=10)
top_edges_after = fun.get_top_weighted_edges(network_graph_after, df_players, shared_minutes, top_n=10)

# --- Weighted in-degree comparison ---
in_degree_comparison = fun.compare_in_degree_before_after(network_graph, playmaker_id, df_players=df_players)

# --- Forward reachability comparison ---
forward_ids = [p['playerId'] for p in lineup_state['lineup'] if p['role'] == 'FW']
reachability_comparison = fun.compare_reachability_before_after(
    network_graph, playmaker_id, forward_ids, df_players=df_players
)

# --- Draw both networks ---
fig_before, ax_before = viz.plot_passing_network(
    network_graph, player_positions, player_weight_totals,
    title=f"{selected_team_row['officialName']} - Passing Network (Intact)"
)

playmaker_name = df_players.loc[df_players['wyId'] == playmaker_id, 'shortName'].iloc[0]

fig_after, ax_after = viz.plot_passing_network_after_removal(
    network_graph_after, positions_after, player_weight_totals,
    playmaker_id, playmaker_position,
    title=f"{selected_team_row['officialName']} - Passing Network (Playmaker Man-Marked)"
)

# --- Your interpretations (edit freely, then re-run this cell to regenerate the report) ---
comments = {
    'intro': "",
    'network': "",
    'efficiency': "",
    'top_edges': "",
    'in_degree': "",
    'reachability': "",
    'conclusion': "",
}

team_slug = selected_team_row['officialName'].replace(' ', '_')
playmaker_slug = playmaker_name.replace(' ', '_')
report_filename = f"report_{team_slug}_{playmaker_slug}.html"

report_path = fun.generate_html_report(
    output_path=os.path.join("report", report_filename),
    team_name=selected_team_row['officialName'],
    playmaker_name=playmaker_name,
    fig_before=fig_before,
    fig_after=fig_after,
    weight_decay=weight_decay,
    in_degree_comparison=in_degree_comparison,
    reachability_comparison=reachability_comparison,
    top_edges_before=top_edges_before,
    top_edges_after=top_edges_after,
    comments=comments,
)
# %%
