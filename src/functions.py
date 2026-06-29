# %% [1] Key Libraries and Custom Modules
# standard libraries
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

_EVENTS_CACHE = {}
_MATCHES_CACHE = {}

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
    print(f"[INFO] Loading event file from disk: {event_file}. Wait before running the next cell.")
 
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

# %% [3] Pass weighting functions
  
 
def zone_value(x, y, exponent =1.5):
    """
    Returns a danger score for a pitch location (x, y). The score is higher the closer and
    more central the location is relative to the opponent's goal.

    The 'exponent' parameter makes the value curve non-linear: with
    exponent > 1, value accelerates as you get closer to the goal.
    """
    # Opponent's goal position in pitch coordinates (0-100 scale, x=100 is the opponent's goal line, y=50 is the center)
    GOAL_X, GOAL_Y = 100, 50
 
    # Maximum possible distance from the goal point.
    MAX_DIST_FROM_GOAL = (GOAL_X ** 2 + GOAL_Y ** 2) ** 0.5

    dist_from_goal = ((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2) ** 0.5
    linear_value = MAX_DIST_FROM_GOAL - dist_from_goal
    normalized = linear_value / MAX_DIST_FROM_GOAL
    return (normalized ** exponent) * MAX_DIST_FROM_GOAL 
 
def compute_pass_weight(start_x, start_y, end_x, end_y, alpha=0.6, beta=0.4, exponent=1.5):
    """
    Computes the weight of a single pass as a combination of:
      - delta_value: how much the pass progresses the ball toward a more
        dangerous zone (end zone value minus start zone value)
      - end_value: how dangerous the zone the ball ends up in is, regardless
        of where it came from.
 
    weight = alpha * delta_value + beta * end_value
 
    A small floor at 0 is applied to avoid backwards passes having negative weights.
    """
    start_value = zone_value(start_x, start_y, exponent=exponent)
    end_value = zone_value(end_x, end_y, exponent=exponent)
    delta_value = end_value - start_value
 
    weight = alpha * delta_value + beta * end_value
    return max(0.0, weight)
 
 
def add_pass_weights(df_passes, alpha=0.6, beta=0.4, exponent=1.5):
    """
    Takes a DataFrame of pass events (as returned by load_and_filter_team_passes,
    which has a 'positions' column with start/end pitch coordinates) and adds
    a 'pass_weight' column computed via compute_pass_weight.
    """
    def _weight_from_row(positions):
        start, end = positions[0], positions[1]
        return compute_pass_weight(
            start['x'], start['y'], end['x'], end['y'],
            alpha=alpha, beta=beta, exponent=exponent
        )
 
    df_passes = df_passes.copy()
    df_passes['pass_weight'] = df_passes['positions'].apply(_weight_from_row)
    return df_passes
 
 
def load_players_data(project_root):
    """
    Loads the players.json file from the data directory and fixes any
    double-escaped unicode characters in player names (same issue as
    teams.json, see fix_double_escaped_unicode).
    """
    players_path = os.path.join(project_root, "data", "Data", "players.json")
    with open(players_path, "r", encoding="utf-8") as f:
        players_data = json.load(f)
 
    df_players = pd.DataFrame(players_data)
    for col in ['shortName', 'firstName', 'lastName']:
        if col in df_players.columns:
            df_players[col] = df_players[col].apply(fix_double_escaped_unicode)
    return df_players
 
 
def get_player_pass_weight_totals(df_passes_weighted, df_players, df_minutes=None, min_minutes=1000):
    """
    Aggregates total pass weight per player (sum of pass_weight across all
    their passes), joined with readable player names from players.json.

    If df_minutes is provided (a DataFrame with columns 'playerId' and
    'minutes_played', as returned by get_season_minutes_played), two extra
    columns are added: 'weight_per_90' and 'passes_per_90', normalizing
    total_weight and pass_count by minutes played (scaled to a 90-minute
    match). This avoids players with more minutes simply ranking higher
    just because they were on the pitch longer.

    When df_minutes is provided, players with fewer than 'min_minutes'
    minutes played are excluded from the result entirely (default: 1000),
    since per-90 figures computed from very few minutes are noisy and not
    representative of a player's real involvement. 
    Set min_minutes=0 to disable this filter.

    Returns a DataFrame sorted from highest to lowest weight_per_90, with columns: 
    'playerId', 'shortName', 'weight_per_90', 'total_weight', 'pass_count', 
    'avg_weight_per_pass', 'minutes_played', 'passes_per_90'.
    """
    totals = (
        df_passes_weighted
        .groupby('playerId')['pass_weight']
        .agg(total_weight='sum', pass_count='count')
        .reset_index()
    )
 
    totals = totals.merge(
        df_players[['wyId', 'shortName']],
        left_on='playerId', right_on='wyId', how='left'
    ).drop(columns=['wyId'])

    totals['avg_weight_per_pass'] = totals['total_weight'] / totals['pass_count']

    if df_minutes is not None:
        totals = totals.merge(df_minutes, on='playerId', how='left')

        # Players with no recorded minutes (shouldn't normally happen if they
        # made passes, but guard against division by zero just in case)
        totals['minutes_played'] = totals['minutes_played'].fillna(0)

        totals = totals[totals['minutes_played'] >= min_minutes].copy()

        safe_minutes = totals['minutes_played'].replace(0, np.nan)

        totals['weight_per_90'] = (totals['total_weight'] / safe_minutes) * 90
        totals['passes_per_90'] = (totals['pass_count'] / safe_minutes) * 90

        totals = totals[[
            'playerId', 'shortName', 'weight_per_90', 'total_weight', 'pass_count', 'avg_weight_per_pass',
            'minutes_played', 'passes_per_90'
        ]]
        totals = totals.sort_values('weight_per_90', ascending=False).reset_index(drop=True)
        totals.index = totals.index + 1
    else:
        totals = totals[['playerId', 'shortName', 'total_weight', 'pass_count', 'avg_weight_per_pass']]
        totals = totals.sort_values('total_weight', ascending=False).reset_index(drop=True)
        totals.index = totals.index + 1

    print(totals)
    return totals


# %% [4] Minutes played functions

def get_match_file_for_team(team_row):
    """
    Returns the matches_*.json file name for a given team, using the same
    area-to-file mapping as get_event_file_for_team (matches and events
    files follow an identical naming convention, e.g. matches_Italy.json
    pairs with events_Italy.json).
    """
    event_file, country_name = get_event_file_for_team(team_row)
    match_file = event_file.replace('events_', 'matches_')
    return match_file, country_name


def load_matches_file(project_root, match_file, use_cache=True):
    """
    Loads a matches_*.json file into a list of match dicts, using an
    in-memory cache to avoid reloading the same file multiple times.
    """
    if use_cache and match_file in _MATCHES_CACHE:
        print(f"[INFO] Using cached version of {match_file}")
        return _MATCHES_CACHE[match_file]

    matches_path = os.path.join(project_root, "data", "Data", "matches", match_file)
    print(f"[INFO] Loading matches file from disk: {match_file}")

    with open(matches_path, "r", encoding="utf-8") as f:
        matches_data = json.load(f)

    if use_cache:
        _MATCHES_CACHE[match_file] = matches_data

    return matches_data


def get_team_matches(matches_data, team_id):
    """
    Filters the full matches list down to only the matches where the given
    team_id appears in teamsData (i.e. the matches that team actually played).
    """
    team_id_str = str(team_id)
    return [m for m in matches_data if team_id_str in m.get('teamsData', {})]


def compute_match_duration(df_events, match_id, first_half_minutes=45):
    """
    Estimates the total duration (in minutes) of a match using its events:
    a fixed 45 minutes for the first half, plus the minute of the last
    recorded event in the second half (to capture second-half stoppage time).

    df_events must contain events from BOTH teams for this match (i.e. the
    full events file for the competition, not a single team's filtered passes).
    """
    match_events = df_events[df_events['matchId'] == match_id]
    second_half_events = match_events[match_events['matchPeriod'] == '2H']

    if second_half_events.empty:
        # Fallback: no second-half events found, just use the regulation length
        return first_half_minutes * 2

    last_event_sec = second_half_events['eventSec'].max()
    second_half_minutes = last_event_sec / 60.0

    return first_half_minutes + second_half_minutes


def compute_minutes_played_in_match(match_data, team_id, match_duration):
    """
    Computes minutes played by each player of the given team in a single
    match, using the match's lineup/bench/substitutions data.

    Returns a dict {playerId: minutes_played}.
    """
    team_data = match_data['teamsData'][str(team_id)]
    formation = team_data.get('formation', {})

    lineup = formation.get('lineup', []) or []
    bench = formation.get('bench', []) or []
    substitutions = formation.get('substitutions', []) or []

    # Build quick lookup: playerId -> minute they were substituted out (if any)
    subbed_out_at = {sub['playerOut']: sub['minute'] for sub in substitutions}
    # Build quick lookup: playerId -> minute they were substituted in (if any)
    subbed_in_at = {sub['playerIn']: sub['minute'] for sub in substitutions}

    minutes_played = {}

    for player in lineup:
        player_id = player['playerId']
        if player_id in subbed_out_at:
            minutes_played[player_id] = subbed_out_at[player_id]
        else:
            minutes_played[player_id] = match_duration

    for player in bench:
        player_id = player['playerId']
        if player_id in subbed_in_at:
            minutes_played[player_id] = match_duration - subbed_in_at[player_id]
        # players on the bench who never came on are left out entirely (0 minutes)

    return minutes_played


def get_season_minutes_played(project_root, team_id, team_row, df_events=None, use_cache=True):
    """
    Computes total minutes played by each player of the given team across
    all matches in the season, by combining matches_*.json (lineups,
    substitutions) with events_*.json (to estimate real match duration).

    If df_events is not provided, it will be loaded internally via
    load_events_file (using the same file the team's passes come from).

    Returns a DataFrame with columns: playerId, minutes_played.
    """
    match_file, country_name = get_match_file_for_team(team_row)
    matches_data = load_matches_file(project_root, match_file, use_cache=use_cache)
    team_matches = get_team_matches(matches_data, team_id)
    print(f"[INFO] Found {len(team_matches)} matches for this team in {match_file}")

    if df_events is None:
        event_file, _ = get_event_file_for_team(team_row)
        df_events = load_events_file(project_root, event_file, use_cache=use_cache)

    total_minutes = {}

    for match_data in team_matches:
        match_id = match_data['wyId']
        match_duration = compute_match_duration(df_events, match_id)
        match_minutes = compute_minutes_played_in_match(match_data, team_id, match_duration)

        for player_id, minutes in match_minutes.items():
            total_minutes[player_id] = total_minutes.get(player_id, 0) + minutes

    df_minutes = pd.DataFrame(
        list(total_minutes.items()), columns=['playerId', 'minutes_played']
    )
    return df_minutes


# %% [5] Playmaker-centered lineup functions

def get_player_role_map(df_players):
    """
    Builds a {playerId: role_code} lookup from players.json, using the
    simplified Wyscout role code (code2: 'GK', 'DF', 'MD', 'FW').
    """
    role_map = {}
    for _, player in df_players.iterrows():
        role_info = player.get('role', {}) or {}
        role_map[player['wyId']] = role_info.get('code2', 'UNK')
    return role_map


def get_matches_where_player_started(team_matches, team_id, player_id):
    """
    Filters a list of team matches (as returned by get_team_matches) down to
    only those where the given player_id appears in the starting lineup
    (not just on the bench) for the given team_id.
    """
    starting_matches = []
    for match_data in team_matches:
        team_data = match_data['teamsData'][str(team_id)]
        lineup = team_data.get('formation', {}).get('lineup', []) or []
        lineup_ids = {p['playerId'] for p in lineup}
        if player_id in lineup_ids:
            starting_matches.append(match_data)
    return starting_matches


def get_typical_formation_shape(starting_matches, team_id, role_map):
    """
    Determines the most common formation "shape" (count of players per role)
    across the given matches, e.g. {'GK': 1, 'DF': 4, 'MD': 4, 'FW': 2}.

    Returns the most frequent shape as a dict, plus a dict of all shapes
    seen with their frequency count (useful for sanity-checking how
    consistent the team's formation actually is).
    """
    from collections import Counter

    shape_counter = Counter()

    for match_data in starting_matches:
        team_data = match_data['teamsData'][str(team_id)]
        lineup = team_data.get('formation', {}).get('lineup', []) or []

        role_counts = Counter()
        for player in lineup:
            role = role_map.get(player['playerId'], 'UNK')
            role_counts[role] += 1

        # Use a sorted tuple of (role, count) pairs as a hashable "shape" key
        shape_key = tuple(sorted(role_counts.items()))
        shape_counter[shape_key] += 1

    if not shape_counter:
        return {}, {}

    most_common_shape_key, _ = shape_counter.most_common(1)[0]
    most_common_shape = dict(most_common_shape_key)

    all_shapes = {shape_key: count for shape_key, count in shape_counter.items()}

    return most_common_shape, all_shapes


def get_teammate_frequencies_by_role(starting_matches, team_id, playmaker_id, role_map):
    """
    Counts, across the given matches (where the playmaker started), how
    often each other player started alongside him, grouped by role.

    Returns a dict {role_code: Counter({playerId: count, ...})}, e.g.
    {'DF': Counter({234: 30, 567: 28, ...}), 'MD': Counter({...}), ...}
    The playmaker himself is excluded from these counts.
    """
    from collections import Counter, defaultdict

    frequencies_by_role = defaultdict(Counter)

    for match_data in starting_matches:
        team_data = match_data['teamsData'][str(team_id)]
        lineup = team_data.get('formation', {}).get('lineup', []) or []

        for player in lineup:
            player_id = player['playerId']
            if player_id == playmaker_id:
                continue
            role = role_map.get(player_id, 'UNK')
            frequencies_by_role[role][player_id] += 1

    return dict(frequencies_by_role)


def build_lineup_around_playmaker(typical_shape, teammate_frequencies, playmaker_id, playmaker_role):
    """
    Selects the most frequent teammates per role to build an 11-player
    lineup around the playmaker, respecting the typical formation shape
    (slots per role) determined by get_typical_formation_shape.

    For the playmaker's own role, one slot is reserved for him, so only
    (slot_count - 1) teammates are picked for that role.

    Returns a list of dicts: [{'playerId': ..., 'role': ...}, ...], with
    the playmaker included. If there aren't enough teammates on record to
    fill a role's slots (e.g. very few matches available), fewer players
    than the typical shape are returned for that role, and a warning is printed.
    """
    lineup = [{'playerId': playmaker_id, 'role': playmaker_role}]

    for role, slot_count in typical_shape.items():
        needed = slot_count - 1 if role == playmaker_role else slot_count

        role_counter = teammate_frequencies.get(role, {})
        top_players = sorted(role_counter.items(), key=lambda kv: -kv[1])[:needed]

        if len(top_players) < needed:
            print(f"[WARNING] Only found {len(top_players)} players for role '{role}', "
                  f"needed {needed}. Formation may not be fully representative.")

        for player_id, _count in top_players:
            lineup.append({'playerId': player_id, 'role': role})

    return lineup


def get_substitute_candidates(df_players, team_id, role, exclude_player_ids, teammate_frequencies):
    """
    Returns a list of (label, playerId) tuples for all players in the squad
    (df_players, filtered by currentTeamId == team_id) who share the given
    role, excluding any player already in exclude_player_ids.

    Sorted by how often they started alongside the playmaker (descending),
    using the counts already computed in teammate_frequencies (see
    get_teammate_frequencies_by_role). Players who never started alongside
    the playmaker are still included, just ranked lowest (count = 0), since
    a player might be the "real" starter for that slot despite limited
    overlap with the playmaker.
    """
    role_counter = teammate_frequencies.get(role, {})

    squad = df_players[
        (df_players['currentTeamId'] == team_id) & (~df_players['wyId'].isin(exclude_player_ids))
    ]

    candidates = []
    for _, player in squad.iterrows():
        player_role = (player.get('role', {}) or {}).get('code2', 'UNK')
        if player_role != role:
            continue
        count = role_counter.get(player['wyId'], 0)
        candidates.append((player['wyId'], player['shortName'], count))

    candidates.sort(key=lambda c: -c[2])

    return [
        (f"{name} (id: {pid}, {count} starts w/ playmaker)", pid)
        for pid, name, count in candidates
    ]


def run_playmaker_lineup_builder(project_root, team_row, df_players, player_weight_totals):
    """
    Runs an interactive playmaker selector (dropdown populated from
    player_weight_totals) and, on button click, builds the typical 11-player
    lineup around the selected playmaker (see build_lineup_around_playmaker).

    After the lineup is built, a review step lets the user substitute any
    player with another candidate of the same role (e.g. to fix cases where
    the automatic frequency-based selection picks two players of the same
    specific position, like two left-backs (Happens for example when selecting
    Juventus' starting 11 with Alex Sandro and Asamoah), since role granularity is only
    GK/DF/MD/FW). Substitutions can be made freely; clicking "Confirm Lineup"
    builds and draws the passing network for the lineup as it stands at
    that moment.

    Returns a state dict that updates itself as the user interacts:
        state['playmaker_id']    -> selected playmaker's playerId
        state['playmaker_role']  -> selected playmaker's role code
        state['lineup']          -> list of {'playerId', 'role'} dicts (11 players),
                                     reflects any manual substitutions made
        state['typical_shape']   -> dict of role -> slot count used to build the lineup
        state['network_graph']   -> built only after "Confirm Lineup" is clicked
        state['player_positions'] -> built only after "Confirm Lineup" is clicked
    """
    state = {
        'playmaker_id': None, 'playmaker_role': None, 'lineup': None,
        'typical_shape': None, 'teammate_frequencies': None, 'starting_matches': None,
        'network_graph': None, 'player_positions': None,
    }

    team_id = team_row['wyId']

    playmaker_dropdown = widgets.Dropdown(
        options=[
            (f"{row.shortName} (id: {row.playerId})", row.playerId)
            for row in player_weight_totals.itertuples()
        ],
        description='Playmaker:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )

    build_lineup_button = widgets.Button(description='Build Lineup', button_style='success')
    lineup_output = widgets.Output()

    # --- Substitution controls (populated/shown only after a lineup exists) ---
    substitute_out_dropdown = widgets.Dropdown(
        description='Replace:', style={'description_width': 'initial'}, layout=widgets.Layout(width='350px')
    )
    substitute_in_dropdown = widgets.Dropdown(
        description='With:', style={'description_width': 'initial'}, layout=widgets.Layout(width='350px')
    )
    substitute_button = widgets.Button(description='Substitute', button_style='warning')
    confirm_button = widgets.Button(description='Confirm Lineup', button_style='success')
    substitution_output = widgets.Output()

    def _player_name(player_id):
        row = df_players.loc[df_players['wyId'] == player_id, 'shortName']
        return row.iloc[0] if not row.empty else 'Unknown'

    def _print_lineup(lineup):
        for p in lineup:
            print(f"  {_player_name(p['playerId'])} (id: {p['playerId']}) - role: {p['role']}")

    def _refresh_substitute_out_options():
        substitute_out_dropdown.options = [
            (f"{_player_name(p['playerId'])} ({p['role']})", p['playerId'])
            for p in state['lineup']
        ]

    def _refresh_substitute_in_options(role):
        current_ids = {p['playerId'] for p in state['lineup']}
        candidates = get_substitute_candidates(
            df_players, team_id, role, current_ids, state['teammate_frequencies']
        )
        substitute_in_dropdown.options = candidates

    def on_build_lineup_clicked(b):
        lineup_output.clear_output()
        substitution_output.clear_output()
        playmaker_id = playmaker_dropdown.value

        with lineup_output:
            role_map = get_player_role_map(df_players)
            playmaker_role = role_map.get(playmaker_id, 'UNK')

            match_file, _ = get_match_file_for_team(team_row)
            matches_data = load_matches_file(project_root, match_file)
            team_matches = get_team_matches(matches_data, team_id)
            starting_matches = get_matches_where_player_started(team_matches, team_id, playmaker_id)

            print(f"[INFO] Playmaker started in {len(starting_matches)} matches.")

            typical_shape, all_shapes = get_typical_formation_shape(starting_matches, team_id, role_map)
            print(f"[INFO] Typical formation shape: {typical_shape}")

            teammate_frequencies = get_teammate_frequencies_by_role(
                starting_matches, team_id, playmaker_id, role_map
            )

            lineup = build_lineup_around_playmaker(
                typical_shape, teammate_frequencies, playmaker_id, playmaker_role
            )

            state['playmaker_id'] = playmaker_id
            state['playmaker_role'] = playmaker_role
            state['lineup'] = lineup
            state['typical_shape'] = typical_shape
            state['teammate_frequencies'] = teammate_frequencies
            state['starting_matches'] = starting_matches

            print("[INFO] Lineup built around the playmaker:")
            _print_lineup(lineup)
            print("\n[INFO] Review the lineup below. If it looks wrong (e.g. two "
                  "players of the same specific position), use the controls to "
                  "Substitute, then click 'Confirm Lineup' to re-build the network.")

        _refresh_substitute_out_options()
        if substitute_out_dropdown.value is not None:
            current_role = next(p['role'] for p in state['lineup'] if p['playerId'] == substitute_out_dropdown.value)
            _refresh_substitute_in_options(current_role)

        display(widgets.VBox([
            widgets.HBox([substitute_out_dropdown, substitute_in_dropdown, substitute_button]),
            confirm_button,
            substitution_output
        ]))

    def on_substitute_out_change(change):
        player_out_id = change['new']
        if player_out_id is None:
            return
        role = next(p['role'] for p in state['lineup'] if p['playerId'] == player_out_id)
        _refresh_substitute_in_options(role)

    def on_substitute_clicked(b):
        substitution_output.clear_output()
        player_out_id = substitute_out_dropdown.value
        player_in_id = substitute_in_dropdown.value

        if player_out_id is None or player_in_id is None:
            with substitution_output:
                print("[ERROR] Select both a player to replace and a substitute.")
            return

        for p in state['lineup']:
            if p['playerId'] == player_out_id:
                p['playerId'] = player_in_id
                break

        _refresh_substitute_out_options()

        with substitution_output:
            print(f"[INFO] Substituted {_player_name(player_out_id)} -> {_player_name(player_in_id)}")
            print("\n[INFO] Updated lineup:")
            _print_lineup(state['lineup'])

    def on_confirm_clicked(b):
        with substitution_output:
            print("\n[INFO] Lineup confirmed. Building passing network...")

            lineup = state['lineup']
            lineup_player_ids = {p['playerId'] for p in lineup}
            match_ids = [m['wyId'] for m in state['starting_matches']]

            event_file, _ = get_event_file_for_team(team_row)
            df_events_full = load_events_file(project_root, event_file)

            df_pass_receivers = get_pass_receivers(
                df_events_full, team_id, match_ids, lineup_player_ids
            )

            network_graph, edge_weights = build_passing_network(df_pass_receivers)
            player_positions = get_average_player_positions(df_pass_receivers)

            state['network_graph'] = network_graph
            state['player_positions'] = player_positions

            import src.visuals as viz
            viz.plot_passing_network(
                network_graph, player_positions, player_weight_totals,
                title=f"{team_row['officialName']} - Passing Network (Playmaker's Typical XI)"
            )
            plt.show()

    build_lineup_button.on_click(on_build_lineup_clicked)
    substitute_out_dropdown.observe(on_substitute_out_change, names='value')
    substitute_button.on_click(on_substitute_clicked)
    confirm_button.on_click(on_confirm_clicked)

    display(widgets.VBox([playmaker_dropdown, build_lineup_button, lineup_output]))

    return state


# %% [6] Passing network graph functions

def get_pass_receivers(df_events, team_id, match_ids, lineup_player_ids):
    """
    Identifies the receiver of each successful pass made by team_id, within
    the given match_ids, using the IMMEDIATE NEXT EVENT IN THE MATCH
    (regardless of team) as a proxy for "who received the ball":
      - if that next event belongs to the opposing team, the pass is
        discarded (possession changed hands - we can't reliably say who on
        our team would have received it)
      - if it belongs to our own team, that player is the receiver

    Only passes where BOTH the passer and the receiver are in
    lineup_player_ids are kept (passes to/from players outside the typical
    lineup, e.g. substitutes, are discarded entirely).

    Returns a DataFrame with columns: matchId, passer_id, receiver_id,
    positions (original columns needed for the network).
    """
    relevant_events = df_events[df_events['matchId'].isin(match_ids)].copy()
    relevant_events = relevant_events.sort_values(['matchId', 'matchPeriod', 'eventSec'])

    records = []

    for match_id, match_events in relevant_events.groupby('matchId'):
        for period, period_events in match_events.groupby('matchPeriod'):
            period_events = period_events.reset_index(drop=True)

            for i in range(len(period_events) - 1):
                event = period_events.iloc[i]

                if event['teamId'] != team_id:
                    continue
                if event['eventName'] != 'Pass':
                    continue
                if not _has_tag(event['tags'], 1801):
                    continue

                next_event = period_events.iloc[i + 1]

                # Possession changed hands immediately after the pass: discard,
                # we cannot reliably identify an intended teammate receiver.
                if next_event['teamId'] != team_id:
                    continue

                passer_id = event['playerId']
                receiver_id = next_event['playerId']

                if passer_id not in lineup_player_ids or receiver_id not in lineup_player_ids:
                    continue

                records.append({
                    'matchId': match_id,
                    'passer_id': passer_id,
                    'receiver_id': receiver_id,
                    'positions': event['positions'],
                })

    return pd.DataFrame(records)


def get_average_player_positions(df_pass_receivers):
    """
    Computes each player's average pitch position (x, y), using the START
    coordinates of every pass they made (as a passer) within the given
    pass-receivers DataFrame (see get_pass_receivers).

    Returns a dict {playerId: (avg_x, avg_y)}, suitable for use as a
    node layout (pos=) in networkx/matplotlib drawing functions.
    """
    positions = {}

    for player_id, group in df_pass_receivers.groupby('passer_id'):
        xs = [row.positions[0]['x'] for row in group.itertuples(index=False)]
        ys = [row.positions[0]['y'] for row in group.itertuples(index=False)]
        positions[player_id] = (np.mean(xs), np.mean(ys))

    return positions


def build_passing_network(df_pass_receivers, alpha=0.6, beta=0.4, exponent=1.5):
    """
    Builds a weighted directed graph (networkx.DiGraph) from a pass-receivers
    DataFrame (see get_pass_receivers): nodes are playerIds, edges go from
    passer to receiver, with edge weight equal to the SUM of compute_pass_weight
    across all passes between that specific pair (consistent with the
    'sum, not average' choice made for player-level totals).

    Returns the graph along with a DataFrame of edge weights (passer_id,
    receiver_id, edge_weight, pass_count) for inspection/debugging.
    """
    df = df_pass_receivers.copy()

    def _weight_from_row(positions):
        start, end = positions[0], positions[1]
        return compute_pass_weight(
            start['x'], start['y'], end['x'], end['y'],
            alpha=alpha, beta=beta, exponent=exponent
        )

    df['pass_weight'] = df['positions'].apply(_weight_from_row)

    edge_weights = (
        df.groupby(['passer_id', 'receiver_id'])['pass_weight']
        .agg(edge_weight='sum', pass_count='count')
        .reset_index()
    )

    graph = nx.DiGraph()
    for row in edge_weights.itertuples(index=False):
        graph.add_edge(row.passer_id, row.receiver_id, weight=row.edge_weight, pass_count=row.pass_count)

    return graph, edge_weights


# %% [7] Network resilience analysis functions
 
def add_distance_attribute(graph):
    """
    Adds a 'distance' edge attribute to a copy of the graph, computed as a
    LINEAR inversion of the 'weight' attribute: distance = max_weight - weight.
 
    This is needed because 'weight' represents connection STRENGTH (higher =
    more dangerous/important pass), while shortest-path algorithms in
    networkx interpret edge weights as a COST to minimize. Without this
    inversion, shortest paths would favor weak/safe passing lanes instead of
    the strong/dangerous ones we actually want to measure reachability through.
 
    The linear (rather than 1/weight) inversion was chosen for interpretability:
    distance directly reflects "how far below the strongest possible
    connection" a given pass is, which is easier to explain to a non-technical
    audience (e.g. a coach) than an inverse-proportional relationship.
 
    Returns a new graph (does not modify the input graph in place).
    """
    graph = graph.copy()
    weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    max_weight = max(weights) if weights else 1.0
 
    for u, v, data in graph.edges(data=True):
        data['distance'] = max_weight - data['weight']
 
    return graph
 
 
def compute_global_efficiency(graph):
    """
    Computes the Global Efficiency (Latora & Marchiori) of a weighted graph:
    the average of 1/shortest_distance across all ordered pairs of nodes,
    using the 'distance' edge attribute (see add_distance_attribute).
 
    Disconnected pairs (no path between them) contribute 0 to the average
    (1/infinity = 0), rather than breaking the calculation - this is the
    standard, well-defined behavior of this metric, and is exactly what we
    want: a network that becomes disconnected after removing a node should
    show a clear drop in efficiency, not an error.
 
    Returns a single float.
    """
    graph_with_distance = add_distance_attribute(graph)
    return nx.global_efficiency(graph_with_distance.to_undirected())
 
 
def simulate_playmaker_removal(graph, playmaker_id):
    """
    Removes the playmaker node (and all edges touching it) from a copy of
    the graph, simulating a man-marking / targeted network attack.
 
    Returns the resulting graph (does not modify the input graph in place).
    """
    graph = graph.copy()
    if playmaker_id in graph:
        graph.remove_node(playmaker_id)
    return graph
 
 
def compute_efficiency_decay(graph, playmaker_id):
    """
    Computes Global Efficiency before and after removing the playmaker,
    and the percentage decay between the two.
 
    Returns a dict: {
        'efficiency_before': float,
        'efficiency_after': float,
        'percent_decay': float,  # positive = efficiency dropped
    }
    """
    efficiency_before = compute_global_efficiency(graph)
 
    graph_after = simulate_playmaker_removal(graph, playmaker_id)
    efficiency_after = compute_global_efficiency(graph_after)
 
    if efficiency_before > 0:
        percent_decay = ((efficiency_before - efficiency_after) / efficiency_before) * 100
    else:
        percent_decay = 0.0
 
    return {
        'efficiency_before': efficiency_before,
        'efficiency_after': efficiency_after,
        'percent_decay': percent_decay,
    }
 
 
def get_weighted_in_degree(graph, df_players=None):
    """
    Computes the weighted in-degree (total incoming pass_weight) for every
    node in the graph - i.e. how much total "dangerous pass value" each
    player receives from teammates.
 
    If df_players is provided, player names are attached for readability.
 
    Returns a DataFrame sorted from highest to lowest in-degree, with
    columns: playerId, (shortName), in_degree_weight.
    """
    in_degrees = dict(graph.in_degree(weight='weight'))
 
    df = pd.DataFrame(list(in_degrees.items()), columns=['playerId', 'in_degree_weight'])
 
    if df_players is not None:
        df = df.merge(
            df_players[['wyId', 'shortName']], left_on='playerId', right_on='wyId', how='left'
        ).drop(columns=['wyId'])
        df = df[['playerId', 'shortName', 'in_degree_weight']]
 
    df = df.sort_values('in_degree_weight', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df
 
 
def compare_in_degree_before_after(graph, playmaker_id, df_players=None):
    """
    Computes weighted in-degree before and after removing the playmaker,
    and merges them into a single comparison DataFrame.
 
    Returns a DataFrame with columns: playerId, (shortName), in_degree_before,
    in_degree_after, change, percent_change. Sorted by percent_change
    (most negative first - i.e. players who lost the most relative
    involvement when the playmaker is removed).
 
    Note: the playmaker himself is excluded from the result, since he no
    longer exists in the "after" graph.
    """
    in_degree_before = get_weighted_in_degree(graph, df_players=df_players)
    in_degree_before = in_degree_before.rename(columns={'in_degree_weight': 'in_degree_before'})
 
    graph_after = simulate_playmaker_removal(graph, playmaker_id)
    in_degree_after = get_weighted_in_degree(graph_after, df_players=df_players)
    in_degree_after = in_degree_after.rename(columns={'in_degree_weight': 'in_degree_after'})
 
    merge_cols = ['playerId', 'shortName'] if df_players is not None else ['playerId']
    comparison = in_degree_before.merge(
        in_degree_after[['playerId', 'in_degree_after']], on='playerId', how='left'
    )
 
    comparison = comparison[comparison['playerId'] != playmaker_id].copy()
    comparison['in_degree_after'] = comparison['in_degree_after'].fillna(0)
 
    comparison['change'] = comparison['in_degree_after'] - comparison['in_degree_before']
    comparison['percent_change'] = np.where(
        comparison['in_degree_before'] > 0,
        (comparison['change'] / comparison['in_degree_before']) * 100,
        0.0
    )
 
    comparison = comparison.sort_values('percent_change').reset_index(drop=True)
    comparison.index = comparison.index + 1
    return comparison
 
 
def compute_average_path_length_to_targets(graph, target_player_ids):
    """
    Computes, for each target player (e.g. forwards), the average shortest
    PATH LENGTH (using the 'distance' edge attribute, see add_distance_attribute)
    from every other reachable node TO that target.
 
    This measures "how easy is it, on average, for the rest of the team to
    progress the ball toward this player" - lower distance means more
    reachable/central, higher distance (or infinite, if disconnected) means
    more isolated.
 
    Players with no incoming path from anyone (fully unreachable) are
    reported with average_distance = None and is_reachable = False, rather
    than being silently excluded - this is itself a meaningful result worth
    showing to a coach.
 
    Returns a DataFrame with columns: playerId, (shortName), average_distance,
    is_reachable, reachable_from_count.
    """
    graph_with_distance = add_distance_attribute(graph)
 
    records = []
    for target_id in target_player_ids:
        if target_id not in graph_with_distance:
            records.append({
                'playerId': target_id, 'average_distance': None,
                'is_reachable': False, 'reachable_from_count': 0
            })
            continue
 
        # shortest_path_length with target= gives distances FROM every node
        # TO the target, exactly what we want for "reachability toward target"
        lengths = nx.shortest_path_length(graph_with_distance, target=target_id, weight='distance')
        lengths = {source: dist for source, dist in lengths.items() if source != target_id}
 
        if not lengths:
            records.append({
                'playerId': target_id, 'average_distance': None,
                'is_reachable': False, 'reachable_from_count': 0
            })
        else:
            records.append({
                'playerId': target_id,
                'average_distance': np.mean(list(lengths.values())),
                'is_reachable': True,
                'reachable_from_count': len(lengths),
            })
 
    return pd.DataFrame(records)
 
 
def compare_reachability_before_after(graph, playmaker_id, target_player_ids, df_players=None):
    """
    Computes average path length TO each target player (e.g. forwards),
    before and after removing the playmaker, and merges them for comparison.
 
    Returns a DataFrame with columns: playerId, (shortName), distance_before,
    distance_after, reachable_before, reachable_after, change. If a target
    becomes unreachable after removal (reachable_after=False), 'change' is
    left as None rather than a misleading number, since the comparison is
    no longer numerically meaningful (the player went from "reachable" to
    "not reachable at all" - a categorical change, not a quantitative one).
    """
    before = compute_average_path_length_to_targets(graph, target_player_ids)
    before = before.rename(columns={
        'average_distance': 'distance_before', 'is_reachable': 'reachable_before'
    })[['playerId', 'distance_before', 'reachable_before']]
 
    graph_after = simulate_playmaker_removal(graph, playmaker_id)
    after = compute_average_path_length_to_targets(graph_after, target_player_ids)
    after = after.rename(columns={
        'average_distance': 'distance_after', 'is_reachable': 'reachable_after'
    })[['playerId', 'distance_after', 'reachable_after']]
 
    comparison = before.merge(after, on='playerId', how='left')
 
    comparison['change'] = np.where(
        comparison['reachable_before'] & comparison['reachable_after'],
        comparison['distance_after'] - comparison['distance_before'],
        np.nan
    )
 
    if df_players is not None:
        comparison = comparison.merge(
            df_players[['wyId', 'shortName']], left_on='playerId', right_on='wyId', how='left'
        ).drop(columns=['wyId'])
        comparison = comparison[[
            'playerId', 'shortName', 'distance_before', 'distance_after',
            'reachable_before', 'reachable_after', 'change'
        ]]
 
    return comparison
 
 
# %% [7b] Network-level weight decay (alternative to Global Efficiency)
 
def compute_lineup_weight_per_90(df_team_passes_weighted, lineup_player_ids, match_ids, project_root, team_id, team_row):
    """
    Computes weight_per_90 for each player in lineup_player_ids, using ONLY
    the passes and minutes played within the given match_ids (i.e. the
    matches where the playmaker started - see get_matches_where_player_started),
    rather than the player's full-season totals.
 
    This keeps the network-level decay metric consistent with the rest of
    the resilience analysis, which is scoped to that same set of matches.
 
    Returns a DataFrame with columns: playerId, weight_per_90.
    """
    passes_in_matches = df_team_passes_weighted[
        df_team_passes_weighted['matchId'].isin(match_ids) &
        df_team_passes_weighted['playerId'].isin(lineup_player_ids)
    ]
 
    weight_totals = passes_in_matches.groupby('playerId')['pass_weight'].sum()
 
    match_file, _ = get_match_file_for_team(team_row)
    matches_data = load_matches_file(project_root, match_file)
    minutes_in_matches = {}
    for match_data in matches_data:
        if match_data['wyId'] not in match_ids:
            continue
        if str(team_id) not in match_data['teamsData']:
            continue
        match_duration = compute_match_duration(
            load_events_file(project_root, get_event_file_for_team(team_row)[0]),
            match_data['wyId']
        )
        match_minutes = compute_minutes_played_in_match(match_data, team_id, match_duration)
        for player_id, minutes in match_minutes.items():
            minutes_in_matches[player_id] = minutes_in_matches.get(player_id, 0) + minutes
 
    records = []
    for player_id in lineup_player_ids:
        total_weight = weight_totals.get(player_id, 0.0)
        total_minutes = minutes_in_matches.get(player_id, 0)
        weight_per_90 = (total_weight / total_minutes * 90) if total_minutes > 0 else 0.0
        records.append({'playerId': player_id, 'weight_per_90': weight_per_90})
 
    return pd.DataFrame(records)
 
 
def compute_network_weight_decay(lineup_weight_per_90, playmaker_id):
    """
    Computes a simple, easy-to-explain "collective decay" metric: the sum of
    weight_per_90 across all players in the lineup, before and after removing
    the playmaker's own contribution from that sum.
 
    This is offered as an alternative to Global Efficiency, which tends to
    show near-zero decay on small, densely-connected 11-player networks
    (most players remain mutually reachable through alternative routes even
    after the playmaker is removed, so the network-wide average barely moves).
    Summing weight_per_90 instead directly captures how much total
    "dangerous passing output" disappears from the team's accounting when
    that one player's contribution is taken out.
 
    Returns a dict: {'weight_before': float, 'weight_after': float, 'percent_decay': float}
    """
    weight_before = lineup_weight_per_90['weight_per_90'].sum()
 
    playmaker_weight = lineup_weight_per_90.loc[
        lineup_weight_per_90['playerId'] == playmaker_id, 'weight_per_90'
    ]
    playmaker_weight = playmaker_weight.iloc[0] if not playmaker_weight.empty else 0.0
 
    weight_after = weight_before - playmaker_weight
 
    percent_decay = ((weight_before - weight_after) / weight_before * 100) if weight_before > 0 else 0.0
 
    return {
        'weight_before': weight_before,
        'weight_after': weight_after,
        'percent_decay': percent_decay,
    }
 
 
def get_shared_minutes_by_pair(project_root, team_id, team_row, match_ids):
    """
    Computes, for every pair of players who were both on the pitch together
    in at least one of the given matches, the total shared minutes across
    those matches (approximated as the minimum of the two players' minutes
    played in each individual match - i.e. the overlap window in which both
    could plausibly have exchanged a pass).
 
    Returns a dict {(player_a, player_b): shared_minutes}, with player_a
    and player_b always ordered as (min(a,b), max(a,b)) so each pair has a
    single, consistent key regardless of passer/receiver direction.
    """
    match_file, _ = get_match_file_for_team(team_row)
    matches_data = load_matches_file(project_root, match_file)
    event_file, _ = get_event_file_for_team(team_row)
    df_events = load_events_file(project_root, event_file)
 
    shared_minutes = {}
 
    for match_data in matches_data:
        if match_data['wyId'] not in match_ids:
            continue
        if str(team_id) not in match_data['teamsData']:
            continue
 
        match_duration = compute_match_duration(df_events, match_data['wyId'])
        match_minutes = compute_minutes_played_in_match(match_data, team_id, match_duration)
 
        players = list(match_minutes.keys())
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                a, b = players[i], players[j]
                key = (a, b) if a < b else (b, a)
                overlap = min(match_minutes[a], match_minutes[b])
                shared_minutes[key] = shared_minutes.get(key, 0) + overlap
 
    return shared_minutes
 
 
def get_top_weighted_edges(graph, df_players, shared_minutes, top_n=10):
    """
    Returns the top_n passer->receiver pairs with the highest WEIGHT PER 90
    SHARED MINUTES in the graph, with readable player names attached.
 
    Raw edge weight (a simple sum over all matches) would unfairly favor
    pairs of players who simply played more minutes together, regardless of
    how dangerous their specific connection actually is. Dividing by the
    shared minutes between that specific pair (see get_shared_minutes_by_pair)
    and scaling to a 90-minute basis makes the comparison fair across pairs
    with very different amounts of game time together.
 
    Returns a DataFrame with columns: passer_name, receiver_name,
    weight_per_90, total_weight, pass_count, shared_minutes, sorted from
    highest to lowest weight_per_90.
    """
    name_lookup = df_players.set_index('wyId')['shortName'].to_dict()
 
    records = []
    for u, v, data in graph.edges(data=True):
        key = (u, v) if u < v else (v, u)
        minutes = shared_minutes.get(key, 0)
        weight_per_90 = (data['weight'] / minutes * 90) if minutes > 0 else 0.0
 
        records.append({
            'passer_name': name_lookup.get(u, str(u)),
            'receiver_name': name_lookup.get(v, str(v)),
            'weight_per_90': weight_per_90,
            'total_weight': data['weight'],
            'pass_count': data.get('pass_count', None),
            'shared_minutes': minutes,
        })
 
    df = pd.DataFrame(records).sort_values('weight_per_90', ascending=False).head(top_n).reset_index(drop=True)
    df.index = df.index + 1
    return df
 
 
# Center and radius for the "dangerous zone" circle used to identify forward
# receptions worth counting: centered slightly behind the goal line (so the
# circle bulges forward to cover the box and nearby flanks evenly), with a
# radius reaching out to roughly the edge of the final third.
DANGER_ZONE_CENTER = (110, 50)
DANGER_ZONE_RADIUS = 44
 
 
def is_in_danger_zone(x, y):
    """
    Returns True if pitch location (x, y) falls within the "dangerous zone"
    circle (see DANGER_ZONE_CENTER, DANGER_ZONE_RADIUS) used to identify
    meaningful forward receptions.
    """
    cx, cy = DANGER_ZONE_CENTER
    distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    return distance <= DANGER_ZONE_RADIUS
 
 
def get_forward_receptions_in_danger_zone(df_pass_receivers, forward_ids, playmaker_id):
    """
    For each forward in forward_ids, finds every pass they received (as
    receiver_id in df_pass_receivers) that landed inside the "dangerous
    zone" (see is_in_danger_zone), using the pass's END coordinates
    (positions[1]).
 
    Returns a dict {forward_id: {'before': [(x, y), ...], 'after': [(x, y), ...]}}:
      - 'before' includes ALL such receptions, regardless of who passed it
      - 'after' includes only those where the passer was NOT the playmaker
        (i.e. an observational approximation of "receptions that would
        still happen if the playmaker were taken out of the equation")
 
    This is a count of OBSERVED events (not a simulated shortest-path
    metric), so it is robust to the "one lucky long pass" issue that an
    unweighted shortest-path calculation can suffer from.
    """
    results = {fwd_id: {'before': [], 'after': []} for fwd_id in forward_ids}
 
    for row in df_pass_receivers.itertuples(index=False):
        if row.receiver_id not in results:
            continue
 
        end_x, end_y = row.positions[1]['x'], row.positions[1]['y']
        if not is_in_danger_zone(end_x, end_y):
            continue
 
        results[row.receiver_id]['before'].append((end_x, end_y))
        if row.passer_id != playmaker_id:
            results[row.receiver_id]['after'].append((end_x, end_y))
 
    return results
 
 
# %% [8] HTML report generation
 
def generate_html_report(
    output_path,
    team_name,
    playmaker_name,
    fig_before,
    fig_after,
    weight_decay,
    in_degree_comparison,
    fig_danger_zone,
    top_edges_before,
    top_edges_after,
    comments=None,
):
    """
    Assembles a print-friendly, light-themed HTML report combining the
    passing network visuals, the resilience metrics, and the user's own
    written interpretations.
 
    Parameters:
        output_path: where to write the .html file
        team_name, playmaker_name: strings, used in headers
        fig_before, fig_after: matplotlib Figure objects (the intact and
            after-removal network plots, e.g. from plot_passing_network and
            plot_passing_network_after_removal). Embedded as base64 images.
        weight_decay: dict, as returned by compute_network_weight_decay
            (sum of weight_per_90 across the lineup, before/after removing
            the playmaker's own contribution - used instead of Global
            Efficiency, which tends to show near-zero decay on small,
            densely-connected lineup networks)
        in_degree_comparison: DataFrame, as returned by
            compare_in_degree_before_after
        fig_danger_zone: matplotlib Figure, as returned by
            plot_forward_receptions_in_danger_zone (before/after dot plot
            of forward receptions near the opponent's goal). Replaces the
            earlier shortest-path-based reachability metric, which proved
            unstable on small lineup networks (see conversation notes:
            removing a node can spuriously shorten unrelated paths).
        top_edges_before, top_edges_after: DataFrames, as returned by
            get_top_weighted_edges (called once on the intact graph, once
            on the graph after removal)
        comments: optional dict of strings to insert as the "your
            interpretation" text under each section. Recognized keys:
            'intro', 'network', 'efficiency', 'top_edges', 'in_degree',
            'reachability', 'conclusion'. Any missing key is left as an
            empty placeholder you can fill in later by editing the
            generated HTML directly, or by re-running this function with
            an updated dict.
 
    Returns the output_path (so it can be passed straight to a "files to
    open" helper, e.g. present_files in the assistant's own environment).
    """
    comments = comments or {}
 
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
 
    def _fig_to_base64(fig):
        """Encodes a matplotlib figure as a base64 PNG string, embeddable
        directly in an <img> tag without writing a separate file to disk."""
        import io
        import base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return encoded
 
    before_img_base64 = _fig_to_base64(fig_before)
    after_img_base64 = _fig_to_base64(fig_after)
 
    def _comment_block(key):
        text = comments.get(key, '').strip()
        if not text:
            text = '<em>[Your interpretation goes here.]</em>'
        return f'<div class="comment-box">{text}</div>'
 
    def _format_df(df, float_cols=None, bool_cols=None):
        df = df.copy()
        float_cols = float_cols or []
        bool_cols = bool_cols or []
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map(lambda v: "Yes" if v else "No")
        return df.to_html(index=False, classes='data-table', border=0, na_rep="—")
 
    in_degree_html = _format_df(
        in_degree_comparison,
        float_cols=['in_degree_before', 'in_degree_after', 'change', 'percent_change']
    )
    danger_zone_img_base64 = _fig_to_base64(fig_danger_zone)
    top_edges_before_html = _format_df(
        top_edges_before, float_cols=['weight_per_90', 'total_weight']
    )
    top_edges_after_html = _format_df(
        top_edges_after, float_cols=['weight_per_90', 'total_weight']
    )
 
    decay_sign = "decrease" if weight_decay['percent_decay'] > 0 else "increase"
 
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tactical Vulnerability Report - {team_name}</title>
<style>
    @media print {{
        body {{ margin: 0; }}
        section {{ page-break-inside: avoid; }}
    }}
    body {{
        font-family: 'Georgia', 'Times New Roman', serif;
        max-width: 880px;
        margin: 0 auto;
        padding: 40px 24px 80px;
        color: #1a1a1a;
        background: #ffffff;
        line-height: 1.55;
    }}
    h1 {{
        font-size: 28px;
        border-bottom: 3px solid #1a1a1a;
        padding-bottom: 12px;
        margin-bottom: 4px;
    }}
    .subtitle {{
        color: #555;
        font-size: 15px;
        margin-bottom: 40px;
        font-style: italic;
    }}
    h2 {{
        font-size: 20px;
        margin-top: 56px;
        border-left: 5px solid #b22222;
        padding-left: 12px;
    }}
    .metric-explainer {{
        background: #f7f5f0;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 14px 18px;
        font-size: 14.5px;
        color: #333;
        margin: 14px 0 18px;
    }}
    .comment-box {{
        background: #fffbea;
        border-left: 4px solid #d4a017;
        padding: 12px 18px;
        margin: 14px 0 30px;
        font-size: 15px;
        color: #3a3a3a;
    }}
    .network-images {{
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        justify-content: center;
        margin: 20px 0;
    }}
    .network-images img {{
        width: 100%;
        max-width: 420px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }}
    .image-caption {{
        text-align: center;
        font-size: 13px;
        color: #666;
        margin-top: 4px;
    }}
    .big-numbers {{
        display: flex;
        gap: 24px;
        justify-content: center;
        margin: 24px 0;
        flex-wrap: wrap;
    }}
    .big-number-card {{
        text-align: center;
        padding: 16px 28px;
        border: 1px solid #ddd;
        border-radius: 6px;
        min-width: 140px;
    }}
    .big-number-card .value {{
        font-size: 30px;
        font-weight: bold;
    }}
    .big-number-card .label {{
        font-size: 13px;
        color: #666;
        margin-top: 4px;
    }}
    .decay-card .value {{ color: #b22222; }}
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13.5px;
        margin: 12px 0 24px;
    }}
    table.data-table th {{
        background: #1a1a1a;
        color: white;
        padding: 8px 10px;
        text-align: left;
    }}
    table.data-table td {{
        padding: 7px 10px;
        border-bottom: 1px solid #e0e0e0;
    }}
    table.data-table tr:nth-child(even) {{
        background: #f7f5f0;
    }}
    footer {{
        margin-top: 60px;
        font-size: 12px;
        color: #999;
        text-align: center;
        border-top: 1px solid #ddd;
        padding-top: 16px;
    }}
</style>
</head>
<body>
 
<h1>Tactical Vulnerability Report</h1>
<div class="subtitle">
    {team_name} — Isolating playmaker: <strong>{playmaker_name}</strong>
</div>
 
{_comment_block('intro')}
 
<h2>1. Passing Network — Intact vs. After Man-Marking</h2>
<div class="metric-explainer">
    The graphs below show the team's typical passing structure on the pitch.
    Players' node color and size reflects their <strong>weight per 90 minutes</strong>
    (how much they are involved in the team's overall ball progression and possession).
    Edge thickness reflects how frequent that specific passing connection is.
    On the right, the playmaker has been removed (man-marked).
</div>
<div class="network-images">
    <div>
        <img src="data:image/png;base64,{before_img_base64}" alt="Network before removal">
        <div class="image-caption">Before — full network</div>
    </div>
    <div>
        <img src="data:image/png;base64,{after_img_base64}" alt="Network after removal">
        <div class="image-caption">After — playmaker man-marked</div>
    </div>
</div>
{_comment_block('network')}
 
<h2>2. Collective Passing Output Decay</h2>
<div class="metric-explainer">
    This metric sums each lineup player's <strong>weight per 90 minutes</strong>
    (their individual rate of dangerous, progressive passing involvement,
    computed specifically across the matches where the playmaker started)
    to get a total "collective passing output" for the team. The "after"
    value simply removes the playmaker's own contribution from that sum —
    showing how much total dangerous passing output disappears from the
    team's accounting once he's taken out, independent of how the rest of
    the network reorganizes around the gap.
</div>
<div class="big-numbers">
    <div class="big-number-card">
        <div class="value">{weight_decay['weight_before']:.1f}</div>
        <div class="label">Collective output — Before</div>
    </div>
    <div class="big-number-card">
        <div class="value">{weight_decay['weight_after']:.1f}</div>
        <div class="label">Collective output — After</div>
    </div>
    <div class="big-number-card decay-card">
        <div class="value">{weight_decay['percent_decay']:.1f}%</div>
        <div class="label">{decay_sign} in output</div>
    </div>
</div>
{_comment_block('efficiency')}
 
<h2>3. Most Dangerous Passing Lanes — Before vs. After</h2>
<div class="metric-explainer">
    These tables list the strongest individual passer→receiver connections
    in the network (by edge weight per 90 mins), before and after the playmaker
    is removed. Comparing the two lists shows which dangerous lanes
    disappear entirely, and which ones emerge to take their place once the team 
    has to circulate the ball without him.
</div>
<div>
    <strong>Before</strong>
    {top_edges_before_html}
    <strong>After</strong>
    {top_edges_after_html}
</div>
{_comment_block('top_edges')}
 
<h2>4. Who Loses the Most Involvement? (Weighted In-Degree)</h2>
<div class="metric-explainer">
    This table shows how much total "dangerous pass value" each player
    receives from teammates, before and after removing the playmaker.
    A large negative percent change means that player became much less
    involved in the team's passing once the playmaker was taken out —
    i.e. he depended heavily on that specific connection.
</div>
{in_degree_html}
{_comment_block('in_degree')}
 
<h2>5. Forward Receptions in the Dangerous Zone</h2>
<div class="metric-explainer">
    Each dot is a pass a forward received inside the dashed circle near the
    opponent's goal — an observed count, not a simulated metric. 
    "After" removes every reception that came directly
    from the playmaker, showing how many dangerous touches each forward
    would lose if that specific source was cut off.
</div>
<div class="network-images">
    <img src="data:image/png;base64,{danger_zone_img_base64}" alt="Forward receptions in dangerous zone" style="max-width:100%;">
</div>
{_comment_block('reachability')}
 
<h2>6. Conclusion</h2>
{_comment_block('conclusion')}
 
<footer>
    Generated automatically — Sports Data Science Final Project, Livio Guerra (s4444159), Leiden University.
</footer>
 
</body>
</html>
"""
 
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
 
    print(f"[INFO] Report written to {output_path}")
    return output_path
