# %% [1] Key Libraries and Custom Modules
# standard libraries
import os
import json
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

_EVENTS_CACHE = {}

# %% [2] Loading and Filtering Functions

def fix_double_escaped_unicode(text):
    """
    Fixes double-escaped unicode characters in a string.
    """
    if not isinstance(text, str):
        return text
    if "\\u" not in text:
        return text
    try:
        return text.encode('utf-8').decode('unicode_escape')
    except UnicodeDecodeError:
        return text
 
 
def load_teams_data(project_root):
    """
    Loads the teams.json file from the data directory and fixes any
    double-escaped unicode characters in the official team names.
    """
    teams_path = os.path.join(project_root, "data", "Data", "teams.json")
    with open(teams_path, "r", encoding="utf-8") as f:
        teams_data = json.load(f)
 
    df_teams = pd.DataFrame(teams_data)
    df_teams['officialName'] = df_teams['officialName'].apply(fix_double_escaped_unicode)
    return df_teams
 
 
def filter_teams_by_keyword(df_teams, keyword):
    """
    Filters teams whose official name contains the user-specified keyword.
    Returns a DataFrame with 'wyId', 'officialName' and 'area'.
    """
    if not keyword:
        return df_teams.iloc[0:0][['wyId', 'officialName', 'area']].copy()
 
    filtered = df_teams[df_teams['officialName'].str.contains(keyword, case=False, na=False)]
    return filtered[['wyId', 'officialName', 'area']].copy()
 
 
def get_event_file_for_team(team_row):
    """
    Determines the correct events_*.json file to load based on the team's country area.
    """
    area_info = team_row.get('area', {}) or {}
    country_name = area_info.get('name', '')
 
    # Mapping dictionary based exactly on your physical files in data/Data/events/
    file_mapping = {
        'Italy': 'events_Italy.json',
        'England': 'events_England.json',
        'Spain': 'events_Spain.json',
        'France': 'events_France.json',
        'Germany': 'events_Germany.json',
        'European Championship': 'events_European_Championship.json',
        'World Cup': 'events_World_Cup.json'
    }
 
    return file_mapping.get(country_name, 'events_Italy.json'), country_name
 
 
def load_events_file(project_root, event_file, use_cache=True):
    """
    Loads an events_*.json file into a DataFrame, using an in-memory cache
    to avoid reloading heavy files multiple times in the same session.
    """
    if use_cache and event_file in _EVENTS_CACHE:
        print(f"[INFO] Using cached version of {event_file}")
        return _EVENTS_CACHE[event_file]
 
    events_path = os.path.join(project_root, "data", "Data", "events", event_file)
    print(f"[INFO] Loading event file from disk: {event_file}")
 
    if not os.path.exists(events_path):
        print(f"[ERROR] File not found at {events_path}. Falling back to events_Italy.json")
        event_file = "events_Italy.json"
        events_path = os.path.join(project_root, "data", "Data", "events", event_file)
        if use_cache and event_file in _EVENTS_CACHE:
            return _EVENTS_CACHE[event_file]
 
    with open(events_path, "r", encoding="utf-8") as f:
        events_data = json.load(f)
 
    df_events = pd.DataFrame(events_data)
 
    if use_cache:
        _EVENTS_CACHE[event_file] = df_events
 
    return df_events
 
 
def _has_tag(tags, tag_id):
    """
    Robust tag checking for an event: handles the case where 'tags'
    is not a valid list (NaN, None) or contains dictionaries without the 'id' key.
    """
    if not isinstance(tags, list):
        return False
    return any(isinstance(tag, dict) and tag.get('id') == tag_id for tag in tags)
 
 
def load_and_filter_team_passes(project_root, team_id, team_row, use_cache=True):
    """
    Dynamically identifies the correct league or tournament events file
    based on the team's country area, loads it (using cache when possible)
    and filters successful passes for the given team.
    """
    event_file, country_name = get_event_file_for_team(team_row)
    print(f"[INFO] Team area '{country_name}' -> event file: {event_file}")
 
    df_events = load_events_file(project_root, event_file, use_cache=use_cache)
 
    df_team_passes = df_events[
        (df_events['teamId'] == team_id) &
        (df_events['eventName'] == 'Pass') &
        (df_events['tags'].apply(lambda x: _has_tag(x, 1801)))
    ].copy()
 
    df_team_passes = df_team_passes.sort_index()
    return df_team_passes
 
 
def run_team_selector(project_root, df_teams):
    """
    Runs an interactive team selector using ipywidgets. Allows the user to search
    for a team by keyword, select from the results, and load the corresponding
    successful passes for that team. Returns a state dictionary containing the
    selected team row and the filtered passes DataFrame.
    """
    state = {'selected_team_row': None, 'df_team_passes': None}
 
    search_box = widgets.Text(
        value='',
        placeholder='Ex. Arsenal, Roma, Bayern...',
        description='Team:',
        style={'description_width': 'initial'}
    )
 
    team_dropdown = widgets.Dropdown(
        options=[],
        description='Results:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
 
    confirm_button = widgets.Button(
        description='Load Team Passes',
        button_style='success',
        disabled=True
    )
 
    output_area = widgets.Output()
 
    def on_search_change(change):
        keyword = change['new'].strip()
        output_area.clear_output()
        confirm_button.disabled = True
 
        if len(keyword) < 2:
            team_dropdown.options = []
            return
 
        matches = filter_teams_by_keyword(df_teams, keyword)
 
        if matches.empty:
            team_dropdown.options = []
            with output_area:
                print(f"[INFO] No team found for '{keyword}'.")
            return
 
        options = [
            (f"{row.officialName} (id: {row.wyId})", row.wyId)
            for row in matches.itertuples()
        ]
        team_dropdown.options = options
        confirm_button.disabled = False
 
    def on_confirm_clicked(b):
        output_area.clear_output()
        selected_id = team_dropdown.value
 
        if selected_id is None:
            with output_area:
                print("[ERROR] No team selected.")
            return
 
        selected_team_row = df_teams[df_teams['wyId'] == selected_id].iloc[0]
        state['selected_team_row'] = selected_team_row
 
        with output_area:
            print(f"[INFO] Selected team: {selected_team_row['officialName']} (id: {selected_id})")
            df_team_passes = load_and_filter_team_passes(project_root, selected_id, selected_team_row)
            state['df_team_passes'] = df_team_passes
            print(f"[INFO] Loaded {len(df_team_passes)} successful passes for this team.")
            display(df_team_passes.head())
            print("Selected team passes dataframe loaded!")
 
    search_box.observe(on_search_change, names='value')
    confirm_button.on_click(on_confirm_clicked)
 
    display(widgets.VBox([search_box, team_dropdown, confirm_button, output_area]))
 
    return state
