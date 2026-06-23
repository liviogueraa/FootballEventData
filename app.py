# %% [markdown]
# # Main Pipeline - Sports Data Science Final Project
# **Livio Guerra** s4444159

# Quantification of the Tactical impact of man-marking and isolating playmakers through a Network Resilience Analysis. 

# %% [1] Load Key Libraries and Custom Modules
# standard libraries
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
 
# data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch, VerticalPitch
 
# custom modules
from src.functions import load_teams_data, run_team_selector
 
print("All libraries imported successfully!")


# %% [2] Load teams data

PROJECT_ROOT = os.getcwd()
df_teams = load_teams_data(PROJECT_ROOT)

team_state = run_team_selector(PROJECT_ROOT, df_teams)

# %% [3] 
