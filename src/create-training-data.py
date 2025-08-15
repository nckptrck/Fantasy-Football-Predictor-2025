import pandas as pd

# LOAD NFL DATA
# run get-data.py first!
weekly = pd.read_csv('data/weekly.csv')
schedules = pd.read_csv('data/schedules.csv')
weekly_stats = pd.read_csv('data/team_weekly_stats.csv')
player_list = pd.read_csv('data/player_list.csv')
rosters = pd.read_csv('data/rosters.csv')


# Function: get_player_data()
# input: player name, season, week, if the row is for a prediction
# ouput: position, avg data for the season, last 3 games, and last game
# note: if week < 4, data will be (at least partially) based on previous season
# note: if building the training set, use for_prediction = False
def get_player_data(player_name:str,season:int,week:int, for_prediction:bool):
    # grab weekly player data
    if 1 <= week <= 3:
        player = weekly[
            (weekly['player_display_name'] == player_name) & 
            ((weekly['season'] == season) | (weekly['season'] == (season - 1)))]

        player = player[~((player['season'] == season) & (player['week'] >= week))]
    elif 4 <= week <= 17:
        player = weekly[
            (weekly['player_display_name'] == player_name) & 
            (weekly['season'] == season) & 
            (weekly['week'] < week)]
    else:
        print('Not a valid fantasy week!')
        return None

    if player.empty:
        return None


    # grab position and team
    # for building training set
    if not for_prediction:
        position = player['position'].iloc[len(player) - 1]
        this_week = weekly[
                (weekly['player_display_name'] == player_name) & 
                (weekly['season'] == season) & 
                (weekly['week'] == week)]
        # check to make sure they played
        if this_week.empty:
            return None

        team = this_week['recent_team'].iloc[0]
    # for making new predictions
    else:
        roster = rosters[
                (rosters['player_name'] == player_name) & 
                (rosters['season'] == season) & 
                (rosters['week'] == week)]
        
        # check to make sure they played
        if roster.empty:
            roster = rosters[
                (rosters['player_name'] == player_name) & 
                (rosters['season'] <= season) & 
                (rosters['week'] < week)]
            if roster.empty:
                return None
            roster = roster.sort_values(by = ["season", 'week'],ascending=True)

        team = roster['team'].iloc[0]
        position = roster['position'].iloc[0]

    # average data across columns
    cols_to_avg = ['passing_yards', 'passing_tds',
                'interceptions', 'passing_epa',
                'carries', 'rushing_yards', 
                'rushing_tds', 'fumbles',
                'fumbles_lost', 'rushing_epa',
                'receptions', 'targets',
                'receiving_yards', 'receiving_tds',
                'receiving_air_yards','receiving_yards_after_catch',
                'receiving_epa', 'target_share',
                 'air_yards_share', 'fantasy_points_ppr',
                'first_downs']

    player_avg = player[cols_to_avg].mean()

    # get avg data for last 3 games
    player_last_3 = player.tail(3)

    player_last_3 = player_last_3[cols_to_avg].mean()

    # get last games data
    player_last_1 = player[cols_to_avg].tail(1)

    # combine
    # fix column names
    player_avg.index = [f'{col}_avg' for col in player_avg.index]
    player_last_3.index = [f'{col}_last3' for col in player_last_3.index]
    player_last_1 = player_last_1.rename(lambda col: f'{col}_last1', axis=1).iloc[0]

    # Add name and position
    id_info = pd.Series({'player_name': player_name,
                        'position': position,
                        'team': team,
                        'week': week,
                        'season': season})

    # Combine all pieces into one series, then make a dataframe
    row = pd.concat([id_info, player_avg, player_last_3, player_last_1])
    player_data = row.to_frame().T 

    return player_data


# Function: get_team_data()
# input: team, week, season
# output: team offensive avgs for season, last 3, and last game
# note: if week < 4, data will be (at least partially) based on previous season
def get_team_data(team:str, season:int, week:int):
    # grab weekly team data

    if 1 <= week <= 3:
        team_off = weekly_stats[
            (weekly_stats['posteam'] == team) & 
            ((weekly_stats['season'] == season) | (weekly_stats['season'] == (season - 1)))]

        team_off = team_off[~((team_off['season'] == season) & (team_off['week'] >= week))]

        team_games = schedules[
            ((schedules['home_team'] == team) | (schedules['away_team'] == team)) &
            ((schedules['season'] == season) | (schedules['season'] == (season -1)))]
            
        team_games = team_games[~((team_games['season'] == season) & (team_games['week'] >= week))]
        
    elif 4 <= week <= 17:
        team_off = weekly_stats[
            (weekly_stats['posteam'] == team) & 
            (weekly_stats['season'] == season) & 
            (weekly_stats['week'] < week)]
        
        # grab point totals
        team_games = schedules[
            ((schedules['home_team'] == team) | (schedules['away_team'] == team)) &
            (schedules['season'] == season) &
            (schedules['week'] < week)
    ]
    else:
        print('Not a valid fantasy week!')
        return None
    # season average
    team_off_avg = (
    team_off.groupby('posteam').
    agg(team_passing_yards_avg = ('team_passing_yards', 'mean'),
         team_rushing_yards_avg = ('team_rushing_yards', 'mean'),
         team_passing_tds_avg = ('team_passing_tds', 'mean'),
         team_rushing_tds_avg = ('team_rushing_tds', 'mean'),
         team_total_plays_avg = ('team_total_plays', 'mean'),
         team_first_downs_avg = ('team_first_downs', 'mean'),
         team_yards_per_play_avg = ('team_yards_per_play', 'mean'),
         team_turnovers_avg = ('team_turnovers', 'mean'))
    ).reset_index()

    # last 3 game avg
    team_off_last3 = (
    team_off.tail(3).groupby('posteam').
    agg(team_passing_yards_last3 = ('team_passing_yards', 'mean'),
         team_rushing_yards_last3 = ('team_rushing_yards', 'mean'),
         team_passing_tds_last3 = ('team_passing_tds', 'mean'),
         team_rushing_tds_last3 = ('team_rushing_tds', 'mean'),
         team_total_plays_last3 = ('team_total_plays', 'mean'),
         team_first_downs_last3 = ('team_first_downs', 'mean'),
         team_yards_per_play_last3 = ('team_yards_per_play', 'mean'),
         team_turnovers_last3 = ('team_turnovers', 'mean'))
    ).reset_index() 

    # last game
    team_off_last1 = (
        team_off.tail(1).
        rename(lambda col: f'{col}_last1', axis=1).
        drop(['game_id_last1','defteam_last1','season_last1','week_last1'], axis = 1).
        rename({'posteam_last1': 'posteam'}, axis = 1)
        )

    # merge all with final row
    team_data = (team_off_avg.
                      merge(team_off_last3,
                      on=['posteam']).
                      merge(team_off_last1,
                      on=['posteam']))

    team_data = team_data.drop(columns=team_data.filter(regex="^posteam").columns)

    team_games = team_games.copy()
    team_games['team_points'] = team_games.apply(
        lambda row: row['home_score'] if row['home_team'] == team else row['away_score'], axis=1
    )

    # add to final row
    team_data['team_points_avg'] = team_games['team_points'].mean()
    team_data['team_points_last1'] = team_games['team_points'].tail(1)
    team_data['team_points_last3'] = team_games['team_points'].tail(3).mean()

    return team_data


# Function: get_opp_data()
# input: opponent, week, season
# output: opponent DEF avg data for season, last 3, and last game
# note: if week < 4, data will be (at least partially) based on previous season
def get_opp_data(opponent:str, season:int, week:int):
    
    if 1 <= week <= 3:
        opponent_old = None
        if opponent == 'LV' and season == 2020:
            opponent_old = 'OAK'

        opponent_weekly = weekly_stats[
            ((weekly_stats['defteam'] == opponent) | (weekly_stats['defteam'] == opponent_old))  & 
            ((weekly_stats['season'] == season) | (weekly_stats['season'] == (season - 1)))]

        opponent_weekly = opponent_weekly[~((opponent_weekly['season'] == season) & (opponent_weekly['week'] >= week))]

        opp_games = schedules[
            (((schedules['home_team'] == opponent) | (schedules['away_team'] == opponent)) | 
            ((schedules['home_team'] == opponent_old) | (schedules['away_team'] == opponent_old))) &
            ((schedules['season'] == season) | (schedules['season'] == (season -1)))]

        opp_games = opp_games[~((opp_games['season'] == season) & (opp_games['week'] >= week))]

    elif 4 <= week <= 17:
        opponent_weekly = weekly_stats[
        (weekly_stats['defteam'] == opponent) & 
        (weekly_stats['season'] == season) & 
        (weekly_stats['week'] < week)]

        # grab point totals
        opp_games = schedules[
        ((schedules['home_team'] == opponent) | (schedules['away_team'] == opponent)) &
        (schedules['season'] == season) &
        (schedules['week'] < week)
        ]
    else:
        print('Not a valid fantasy week!')
        return None

    # season long avgs
    opp_def_avg = (
    opponent_weekly.groupby('defteam').
    agg(opp_passing_yards_avg = ('team_passing_yards', 'mean'),
         opp_rushing_yards_avg = ('team_rushing_yards', 'mean'),
         opp_passing_tds_avg = ('team_passing_tds', 'mean'),
         opp_rushing_tds_avg = ('team_rushing_tds', 'mean'),
         opp_total_plays_avg = ('team_total_plays', 'mean'),
         opp_first_downs_avg = ('team_first_downs', 'mean'),
         opp_yards_per_play_avg = ('team_yards_per_play', 'mean'),
         opp_turnovers_avg = ('team_turnovers', 'mean'))
    ).reset_index()

    # last 3 games
    opp_def_last3 = (
    opponent_weekly.tail(3).groupby('defteam').
    agg(opp_passing_yards_last3 = ('team_passing_yards', 'mean'),
         opp_rushing_yards_last3= ('team_rushing_yards', 'mean'),
         opp_passing_tds_last3 = ('team_passing_tds', 'mean'),
         opp_rushing_tds_last3 = ('team_rushing_tds', 'mean'),
         opp_total_plays_last3 = ('team_total_plays', 'mean'),
         opp_first_downs_last3 = ('team_first_downs', 'mean'),
         opp_yards_per_play_last3 = ('team_yards_per_play', 'mean'),
         opp_turnovers_last3 = ('team_turnovers', 'mean'))
    ).reset_index()

    # last game
    opp_def_last1 = (opponent_weekly.tail(1).
    rename(lambda col: f'{col}_last1', axis=1).
    drop(['game_id_last1','posteam_last1','season_last1','week_last1'], axis = 1).
    rename({'defteam_last1': 'defteam'}, axis = 1)
    )

    opp_def_last1 = opp_def_last1.rename(columns={col: col.replace('team_', 'opp_') for col in opp_def_last1.columns if col.startswith('team_')})

    opp_def = opp_def_avg.merge(opp_def_last1, on='defteam').merge(opp_def_last3, on='defteam').drop('defteam', axis = 1)


    opp_games = opp_games.copy()
    # points allowed
    opp_games['points_allowed'] = opp_games.apply(
        lambda row: row['away_score'] if row['home_team'] == opponent else row['home_score'], axis=1
    )

    # add to final row 
    opp_def['opp_pts_allowed_avg'] = opp_games['points_allowed'].mean()
    opp_def['opp_pts_allowed_last1'] = opp_games['points_allowed'].tail(1).iloc[0]
    opp_def['opp_pts_allowed_last3'] = opp_games['points_allowed'].tail(3).mean()

    return opp_def


# Function: get_vegas_data()
# input: team, week, season
# output: total line, spread, implied team total
def get_vegas_data(team:str, season:int, week:int):
    vegas_stats = schedules[
    ((schedules['home_team'] == team) | (schedules['away_team'] == team)) &
    (schedules['season'] == season) &
    (schedules['week'] == week)
    ]


    vegas_stats = vegas_stats.copy()
    vegas_stats['is_home'] = 1 if team == vegas_stats['home_team'].iloc[0] else 0


    vegas_stats = vegas_stats[['is_home', 'spread_line', 'total_line']]


    vegas_stats['implied_team_total'] = (vegas_stats['total_line'] + vegas_stats['spread_line'] * (2 * vegas_stats['is_home'] - 1)) / 2

    vegas_stats.reset_index(drop = True, inplace = True)

    return vegas_stats


# Function: get_row()
# input: player name, season, week
# output: row to add to the training data set
# note: if building the training set, use for_prediction = False
def get_row(player_name:str,season:int,week:int, for_prediction:bool = True):
    
    # get player data
    player_data = get_player_data(player_name, season, week, for_prediction)
    
    if player_data is None:
        print(f"[WARNING] No data for {player_name}, season {season}, week {week}")
        return None

    # grab team
    team = player_data['team'].iloc[0]

    # get team data
    team_data = get_team_data(team, season, week)

    # grab opponent
    game = schedules[
    (schedules['season'] == season) & (schedules['week'] == week) &
    ((schedules['home_team'] == team) | (schedules['away_team'] == team))]

    if game['home_team'].iloc[0] == team:
        opponent = game['away_team'].iloc[0]
    else:
        opponent = game['home_team'].iloc[0]

    # get opponent data
    opponent_data = get_opp_data(opponent, season, week)
    # get vegas data
    vegas_data = get_vegas_data(team,season,week)
    # combine all
    final_row_df = pd.concat([player_data,team_data,opponent_data,vegas_data], axis = 1)

    return final_row_df


# Function: get_training_data()
# input: player names, seasons, weeks
# output: full training data 
# calls get_row for every
#  skill player & qb from wk1 2022 - wk17 2024

def get_training_data():
    rows = []
    for idx, row in player_list.iterrows():
        newrow = get_row(row['player_name'], row['season'], row['week'], for_prediction = False)
        if newrow is None:
            print('Row was not added to the training data')
            pass
        rows.append(newrow)
    
    training_data = pd.concat(rows, axis = 0).fillna(0)

    return training_data


# build training set and export to csv
# will take a few minutes to run!!!
# training_data = get_training_data()

# training_data.to_csv('data/training.csv')
