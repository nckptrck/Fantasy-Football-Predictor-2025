import nfl_data_py as nfl

# read in nfl data (will want to run/update weekly to get current player stats)
weekly = nfl.import_weekly_data([2019,2020,2021,2022,2023,2024]) # include 1 year before start of training data
pbp = nfl.import_pbp_data([2019,2020,2021,2022,2023,2024]) # include 1 year before start of training data
schedules = nfl.import_schedules([2019,2020,2021,2022,2023,2024]) # include 1 year before start of training data
rosters = nfl.import_weekly_rosters([2020,2021,2022,2023,2024]) # seasons to include in training

# add columns
weekly['fumbles'] = weekly['rushing_fumbles'] + weekly['receiving_fumbles']
weekly['fumbles_lost'] = weekly['rushing_fumbles_lost'] + weekly['receiving_fumbles_lost'] 
weekly['first_downs'] = weekly['passing_first_downs'] + weekly['rushing_first_downs'] + weekly['receiving_first_downs']
# create team weekly stats and active player list

# weekly stats by team for weeks 1-17 (fantasy football season)
pbp = pbp[(pbp['week'].between(1, 17)) & (pbp['season_type'] == 'REG')]
pbp['turnover'] = pbp['interception'] + pbp['fumble_lost']
weekly_stats = (
    pbp.groupby(['game_id', 'posteam', 'defteam', 'season', 'week'])
    .agg(team_passing_yards = ('passing_yards', 'sum'),
         team_rushing_yards = ('rushing_yards', 'sum'),
         team_passing_tds = ('pass_touchdown', 'sum'),
         team_rushing_tds = ('rush_touchdown', 'sum'),
         team_total_plays = ('posteam', 'count'),
         team_first_downs = ('first_down', 'sum'),
         team_yards_per_play = ('yards_gained', 'mean'),
         team_turnovers = ('turnover', 'sum'))
    ).reset_index()


# list of active skill players by week (to loop through and create training data)
player_list = rosters[((rosters['position'] == 'QB') |
                    (rosters['position'] =='RB') |
                    (rosters['position'] == 'WR') |
                    (rosters['position'] == 'TE')) &
                    (rosters['status'] == 'ACT') & 
                    (rosters['week'] < 18)]

player_list = player_list[['season', 'week', 'team', 'player_name']].sort_values(['season', 'week', 'team', 'player_name'])


# export to csvs
weekly.to_csv("/Users/nicholaspatrick/Desktop/projects/Fantasy-Football-Predictor-2025/data/weekly.csv")
weekly_stats.to_csv("/Users/nicholaspatrick/Desktop/projects/Fantasy-Football-Predictor-2025/data/team_weekly_stats.csv")
schedules.to_csv('/Users/nicholaspatrick/Desktop/projects/Fantasy-Football-Predictor-2025/data/schedules.csv')
player_list.to_csv('/Users/nicholaspatrick/Desktop/projects/Fantasy-Football-Predictor-2025/data/player_list.csv')
rosters.to_csv('/Users/nicholaspatrick/Desktop/projects/Fantasy-Football-Predictor-2025/data/rosters.csv')
