"""
TODO:
- Create CRON job to run automatically
"""
from langchain_core.runnables import RunnableSequence

import pandas as pd
from tabulate import tabulate as table
import sys
import argparse
from espn_api.football import League
from datetime import datetime
import re
import json
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from doritostats import luck_index
import progressbar
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np

# Define dates/year
year = datetime.now().year
month =  datetime.now().month
day = datetime.now().day

# Load environment variables from .env file
load_dotenv()

# Get login credentials for leagues
league_id = os.getenv('league_id')
swid = os.getenv('swid')
espn_s2 = os.getenv('espn_s2')
api_key= os.getenv('OPEN_AI_KEY')

league = League(league_id, year, espn_s2, swid)

# Get NFL week
week = league.nfl_week - 1
# week = 6

print(league, "\n", f'Week {week}')

# Create list of teams
teams = league.teams


def fuzzy_merge(df1, df2, key1, key2, key_team1, key_team2, threshold=90, limit=1):
    """
    Perform fuzzy merge between two DataFrames based on the similarity of the values in key1, key_team1 and key2, key_team2.
    Parameters:
    - df1: First DataFrame.
    - df2: Second DataFrame.
    - key1: Column in df1 to match player name.
    - key2: Column in df2 to match player name.
    - key_team1: Column in df1 to match NFL team.
    - key_team2: Column in df2 to match NFL team.
    - threshold: Similarity threshold (0-100).
    - limit: Maximum number of matches to return per key.
    """
    # Create combined key by concatenating player name and NFL team
    df1['combined_key'] = df1[key1] + " " + df1[key_team1]
    df2['combined_key'] = df2[key2] + " " + df2[key_team2]

    # Apply fuzzy matching on the combined key
    matches = df1['combined_key'].apply(
        lambda x: process.extractOne(x, df2['combined_key'], scorer=fuzz.token_sort_ratio, score_cutoff=threshold))

    df1['Best Match'] = matches.apply(lambda x: x[0] if x is not None else None)
    df1['Match Score'] = matches.apply(lambda x: x[1] if x is not None else None)

    # Merge on the 'Best Match' instead of the original key
    merged_df = pd.merge(df1, df2, left_on='Best Match', right_on='combined_key', how='left')

    # Drop the combined_key columns to keep the DataFrame clean
    merged_df = merged_df.drop(columns=['combined_key_x', 'combined_key_y'])

    return merged_df

def gen_power_rankings(pr_week):
    power_rankings = league.power_rankings(week=pr_week)

    # Extract team names
    extracted_team_names = [(record, re.sub(r'Team\((.*?)\)', r'\1', str(team))) #convert team object to string
        for record, team in power_rankings]

    # Convert to Dataframe
    power_rankings = pd.DataFrame(extracted_team_names, columns=['Power Score','Team'])

    # Switch Score and Team Name cols
    power_rankings_df = power_rankings.reindex(columns=['Team', 'Power Score'])

    # Integrate player values into Power Rankings

    # Load players values for the week
    player_values = pd.read_csv(f'/Users/christiangeer/Fantasy_Sports/football/power_rankings/espn-api-v3/player_values/KTC_values_week{pr_week}.csv')

    # Generate DataFrame of Team Rosters
    league_rosters = []
    for team in league.teams:
        # Get list of player objects for each team
        team_roster = team.roster

        for player in team_roster:
            # Append player name, position and the team that they're on
            league_rosters.append([team.team_name, player.name, player.position, player.proTeam])

    league_rosters_df = pd.DataFrame(league_rosters, columns=['Team','Player','Position', 'NFL_Team'])

    # Remove the number at the end of the 'Pos' values
    player_values['Pos'] = player_values['Pos'].str.extract(r'(\D+)')

    # Filter out defenses and PKs
    league_rosters_filtered = league_rosters_df[~league_rosters_df['Position'].isin(['D/ST', 'PK'])]
    player_values_filtered = player_values[~player_values['Pos'].isin(['DST', 'PK'])]

    # Replace 'Marquise Brown' with 'Hollywood Brown' in the 'Player Name' column
    player_values_filtered['Player Name'].replace('Marquise Brown', 'Hollywood Brown')

    # Only print if the dataframe is not empty and has missing values in specific columns
    if (league_rosters_filtered['Player'].isna().sum() > 0) or (league_rosters_filtered['NFL_Team'].isna().sum() > 0):
        print('\nLeague Rosters Players without Team:\n',
              league_rosters_filtered[
                  (league_rosters_filtered['Player'].isna()) | (league_rosters_filtered['NFL_Team'].isna())])

    if (player_values_filtered['Player Name'].isna().sum() > 0) or (
            player_values_filtered['NFL_Team'].isna().sum() > 0):
        print('\nPlayer Values Players without Team:\n',
              player_values_filtered[
                  (player_values_filtered['Player Name'].isna()) | (player_values_filtered['NFL_Team'].isna())])

    # Perform fuzzy merge using Player Name and NFL Team
    player_values_fuzzy_merged = fuzzy_merge(
        player_values_filtered[['Player Name', 'Pos', 'Value', 'NFL_Team']],
        league_rosters_filtered[['Player', 'Team', 'NFL_Team']],  # Ensure 'Team' is included in the merge
        'Player Name',
        'Player',
        'NFL_Team',
        'NFL_Team',
        threshold=85
    )

    # Select and rename the final columns as 'Player Name', 'Pos', 'Value', and 'Team'
    final_df = player_values_fuzzy_merged[['Player Name', 'Pos', 'Value', 'Team']]

    # Drop duplicates, keeping the first occurrence
    final_df = final_df.drop_duplicates(subset='Player Name', keep='first')
    # print(final_df)

    # Check for rostered players without exact value matches
    if final_df[(final_df['Value'] != 'NaN') & (final_df['Team'] == '')].shape[0] > 0:
        roster_check = final_df[(final_df['Value'] != 'NaN') & (final_df['Team'] == '')]  # Checking for unmatched players
        print(f'\n\tCheck for rostered players without exact matches (Week {pr_week}:\n\n{roster_check}')

    # Count the # of players on each roster to get average player value, reducing bias towards teams with extra IR players
    team_roster_count = final_df.groupby(['Team'], as_index=False).size()

    # Group by 'Team' and 'Position', summing 'Value'
    team_pos_values = final_df.groupby(['Team', 'Pos'], as_index=False)['Value'].sum()

    # Rename columns to keep consistency
    team_pos_values.rename(columns={'Position': 'Pos'}, inplace=True)

    # Group by position to get total team value
    team_values = team_pos_values.groupby(['Team'], as_index=False)['Value'].sum()

    # Merge with roster counts
    team_values = team_values.merge(team_roster_count, on='Team')

    # Divide total values by # of players
    team_values['Value'] = round(team_values['Value'] / team_values['size'], 2)

    # Drop size col
    team_values.drop(columns='size', inplace=True)

    # power_rankings and values
    power_rankings_df = power_rankings_df.merge(team_values, on='Team')

    # Convert 'Power Score' to numeric, forcing errors to NaN
    power_rankings_df['Power Score'] = pd.to_numeric(power_rankings_df['Power Score'].str.replace(',', '').str.replace('$', ''),
                                            errors='coerce')

    # Normalize 'Power Score' and 'Value' using min-max normalization
    power_rankings_df['Power Score Normalized'] = (power_rankings_df['Power Score'] - power_rankings_df['Power Score'].min()) / (
                power_rankings_df['Power Score'].max() - power_rankings_df['Power Score'].min())
    power_rankings_df['Value Normalized'] = (power_rankings_df['Value'] - power_rankings_df['Value'].min()) / (power_rankings_df['Value'].max() - power_rankings_df['Value'].min())

    # Parameters for the weight function to achieve f(1) ~ 0.5 and f(15) ~ .1
    a = 0.5585
    b = -0.1147

    # Calculate the weight for the 'Value' column
    value_weight = round(a * np.exp(b * pr_week), 2)

    # Calculate the weight for 'Power Score'
    power_score_weight = 1 - value_weight

    # Calculate the new power score using the weights
    power_rankings_df['New Power Score'] = (power_rankings_df['Power Score Normalized'] * power_score_weight +
                                            power_rankings_df['Value Normalized'] * value_weight) / \
                                           (power_score_weight + value_weight)

    # Drop the intermediate columns
    power_rankings_df = power_rankings_df.drop(columns=['Power Score Normalized', 'Value Normalized'])

    # Sort by new power score
    power_rankings_df = power_rankings_df.sort_values(by=['New Power Score'], ascending=False)

    # Rename columns for output
    power_rankings_df = power_rankings_df.rename(columns={'Power Score':'Performance Score', 'Value':'KTC Value', 'New Power Score':'Power Score'})

    # Divide Performance score by 100 for readability
    power_rankings_df['KTC Value'] = round(power_rankings_df['KTC Value'] / 100, 2)

    # Multiply power  score by 100, round to whole number
    power_rankings_df['Power Score'] = round(power_rankings_df['Power Score'] * 100, 0)

    # Rank 'Performance Score' in descending order
    power_rankings_df['Performance Rank'] = power_rankings_df['Performance Score'].rank(ascending=False, method='min')

    # Rank 'KTC Value' in descending order
    power_rankings_df['KTC Value Rank'] = power_rankings_df['KTC Value'].rank(ascending=False, method='min')

    # Cast to integers
    power_rankings_df['Performance Rank'] = power_rankings_df['Performance Rank'].astype(int)
    power_rankings_df['KTC Value Rank'] = power_rankings_df['KTC Value Rank'].astype(int)

    # Drop the original 'Performance Score' and 'KTC Value' columns
    power_rankings_df = power_rankings_df.drop(columns=['Performance Score', 'KTC Value'])

    # Set index to start at 1
    power_rankings_df = power_rankings_df.set_axis(range(1, len(power_rankings_df) + 1))

    return power_rankings_df, final_df

def weekly_change(power_rankings, prev_power_rankings):
    diffs = []
    emojis = []

    for team in league.teams:
        # get this week's rank
        tw_rank = power_rankings[power_rankings['Team'] == team.team_name].index.values

        # get last weeks' rank
        lw_rank = prev_power_rankings[prev_power_rankings['Team'] == team.team_name].index.values

        # find the difference between last week to this week
        diff = lw_rank - tw_rank

        # turn into list to iterate over
        diff = int(diff.item())

        # append to the list
        diffs.append([team.team_name, diff])

    # Iterate over the list and modify the integer values
    for element in diffs:
        # Check the second item in each sub-list (the integer)
        if isinstance(element[1], int):
            if element[1] > 0:
                element[1] = f"**<span style=\"color: green;\">⬆️ {abs(element[1])} </span>**"
            elif element[1] < 0:
                element[1] = f"**<span style=\"color: red;\">⬇️ {abs(element[1])} </span>**"
            elif element[1] == 0:
                element[1] = ""  # Make it an empty string for no change

    # convert to a dataframe to join with power rankings
    weekly_change_df = pd.DataFrame(diffs, columns=['Team', 'Weekly Change'])

    # join rankings with weekly change
    power_rankings = pd.merge(power_rankings, weekly_change_df, on='Team', how='outer')

    # Define the desired order of columns
    new_column_order = ['Team', 'Power Score', 'Weekly Change', 'Performance Rank', 'KTC Value Rank']

    # Reorder the columns
    power_rankings = power_rankings[new_column_order]

    # Set index to start at 1
    power_rankings = power_rankings.set_axis(range(1, len(power_rankings) + 1))


    return power_rankings

def gen_playoff_prob():
    # Proj wins and losses for rest of season

    # MONTE CARLO PLAYOFF PROBABILITIES
    # print('\nGenerating Monte Carlo Playoff Probabilities...')

    # number of random season's to simulate
    simulations = 100000
    # weeks in the regular season
    league_weeks = 15
    # number of teams to playoffs
    teams_to_play_off = 4

    """
    team_names:: list of team names. list order is used to
    index home_teams and away_teams

    home_teams, away_teams: list of remaining matchups in the regular season.
    Indexes are based on order from team_names

    current_wins: Integer value represents each team's win count.
    The decimal is used to further order teams based on points for eg 644.8 points would be 0.006448.
    Order needs to be the same as team_names
    """

    # Create dictionary of teams and id number to be fed to monte carlo simulations
    # ['Pat'[1], 'Trevor'[2], 'Billy'[3], 'Jack'[4], 'Travis'[5], 'Lucas'[6], 'Cade'[7], 'Christian'[8]]
    team_dictionary = {'Red Zone  Rockets':1, 'Final Deztination':2, 'Game of  Jones':3, 'Comeback Cardinals':4, 'OC Gang':5, 'Hurts Donit':6, 'Shippin Up To Austin':7, 'Allen & Co.':8}

    # Initialize empty lists to store the names of home and away teams for each week
    home_team_names = []
    away_team_names = []

    # Loop through each week from the current week until the last week of the season
    for this_week in range(week, 16):
        # Create emtpy sets to populate with each weeks home and away teams
        week_home_teams = set()
        week_away_teams = set()

        # Retrieve the scoreboard for the current week, which contains matchups
        week_scoreboard = league.scoreboard(this_week)

        # Iterate through each matchup in the scoreboard for the week
        for matchup in week_scoreboard:
            # Add the home and away teams' names to the set of home/away teams
            week_home_teams.add(matchup.home_team.team_name)
            week_away_teams.add(matchup.away_team.team_name)

        # Append the set of home teams for this week to the list of home team names
        home_team_names.append(week_home_teams)

        # Append the set of away teams for this week to the list of away team names
        away_team_names.append(week_away_teams)

    # Flatten the list of sets and replacing team names with their IDs.
    # Give us a list in order of each weeks home and away teams for the rest of the season
    home_teams = [team_dictionary[team.strip()] for teams_set in home_team_names for team in teams_set]
    away_teams = [team_dictionary[team.strip()] for teams_set in away_team_names for team in teams_set]

    # don't need to do below, taken care of in for loop. Format s wins.totalPointsScored as decimal to 6 places
    # current_wins = [2.010742,3.011697,7.013179,2.010177,6.011863,1.010001,6.012642,5.011502]
    current_wins = []
    for team in league.teams:
        wins = team.wins
        scores = team.scores
        total_points_scored = round(sum(scores), 2) / 100000
        current_wins.append(wins + total_points_scored)

    ###ONLY CONFIGURE THE VALUES ABOVE

    teams = [int(x) for x in range(1, len(league.teams) + 1)]
    weeks_played = (league_weeks) - ((len(home_teams)) / (len(teams) / 2))

    last_playoff_wins = [0] * (league_weeks)
    first_playoff_miss = [0] * (league_weeks)

    import datetime

    begin = datetime.datetime.now()
    import random

    league_size = len(teams)

    games_per_week = int(league_size / 2)
    weeks_to_play = int(league_weeks - weeks_played)
    total_games = int(league_weeks * games_per_week)
    games_left = int(weeks_to_play * games_per_week)

    stats_teams = [0] * (league_size)

    play_off_matrix = [[0 for x in range(teams_to_play_off)] for x in range(league_size)]

    pad = int(games_left)

    avg_wins = [0.0] * teams_to_play_off

    for sims in progressbar.progressbar(range(1, simulations + 1)):
        # create random binary array representing a single season's results
        val = [int(random.getrandbits(1)) for x in range(1, (games_left + 1))]

        empty_teams = [0.0] * league_size

        i = 0
        # assign wins based on 1 or 0 to home or away team
        for x in val:
            if (val[i] == 1):
                empty_teams[home_teams[i] - 1] = empty_teams[home_teams[i] - 1] + 1
            else:
                empty_teams[away_teams[i] - 1] = empty_teams[away_teams[i] - 1] + 1
            i = i + 1

        # add the current wins to the rest of season's results
        empty_teams = [sum(x) for x in zip(empty_teams, current_wins)]

        # sort the teams
        sorted_teams = sorted(empty_teams)

        last_playoff_wins[int(round(sorted_teams[(league_size - teams_to_play_off)], 0)) - 1] = last_playoff_wins[int(round(sorted_teams[(league_size - teams_to_play_off)],0)) - 1] + 1
        first_playoff_miss[int(round(sorted_teams[league_size - (teams_to_play_off + 1)], 0)) - 1] = \
        first_playoff_miss[int(round(sorted_teams[league_size - (teams_to_play_off + 1)], 0)) - 1] + 1

        # pick the teams making the playoffs
        for x in range(1, teams_to_play_off + 1):
            stats_teams[empty_teams.index(sorted_teams[league_size - x])] = stats_teams[empty_teams.index(
                sorted_teams[league_size - x])] + 1
            avg_wins[x - 1] = avg_wins[x - 1] + round(sorted_teams[league_size - x], 0)
            play_off_matrix[empty_teams.index(sorted_teams[league_size - x])][x - 1] = \
            play_off_matrix[empty_teams.index(sorted_teams[league_size - x])][x - 1] + 1

    projections = []

    playSpots = []

    for x in range(1, len(stats_teams) + 1):
        vals = ''
        for y in range(1, teams_to_play_off + 1):
            vals = vals + '\t' + str(round((play_off_matrix[x - 1][y - 1]) / simulations * 100.0, 2))

            playSpots.append(round((play_off_matrix[x - 1][y - 1]) / simulations * 100.0, 2))

        playProb = round((stats_teams[x - 1]) / simulations * 100.0, 2)
        playSpots.insert(0, playProb)
        # print("Vals: ", playSpots)
        projections.append(playSpots)
        playSpots = []
        # print(team_names[x-1]+'\t'+str(round((stats_teams[x-1])/simulations*100.0,2))+vals)
    # print(f'Pre dataframe projections\n{projections}')
    # Convert projections to Pandas Dataframe
    projections = pd.DataFrame(projections)

    # create list of team names
    team_names = []
    teams = league.teams

    for team_obj in teams:
        team_names.append(team_obj.team_name)

    # Insert Team Names to DataFrame
    projections.insert(loc=0, column='Team', value=[team.team_name for team in league.teams])
    projections = projections.set_axis(['Team', 'Playoffs', '1st Seed', '2nd Seed', '3rd Seed', '4th Seed'], axis=1)
    projections = projections.sort_values(by=['Playoffs', '1st Seed', '2nd Seed', '3rd Seed', '4th Seed'], ascending=False)
    # projections[['1st Seed','2nd Seed','3rd Seed', '4th Seed']] = projections[['1st Seed','2nd Seed','3rd Seed', '4th Seed']].astype(str) + "%"
    projections.index = np.arange(1, len(projections) + 1)

    median = projections['Playoffs'].median()

    # bold only the playoff teams
    for index, row in projections.iterrows():
        if row['Playoffs'] > median:
            projections.loc[index, 'Team'] = '**' + str(row['Team']) + '**'
            projections.loc[index, 'Playoffs'] = '**' + str(row['Playoffs']) + '%**'
            projections.loc[index, '1st Seed'] = '**' + str(row['1st Seed']) + '%**'
            projections.loc[index, '2nd Seed'] = '**' + str(row['2nd Seed']) + '%**'
            projections.loc[index, '3rd Seed'] = '**' + str(row['3rd Seed']) + '%**'
            projections.loc[index, '4th Seed'] = '**' + str(row['4th Seed']) + '%**'
        else:
            projections.loc[index, 'Playoffs'] = str(row['Playoffs']) + '%'
            projections.loc[index, '1st Seed'] = str(row['1st Seed']) + '%'
            projections.loc[index, '2nd Seed'] = str(row['2nd Seed']) + '%'
            projections.loc[index, '3rd Seed'] = str(row['3rd Seed']) + '%'
            projections.loc[index, '4th Seed'] = str(row['4th Seed']) + '%'

    print('')

    # print('Average # of wins for playoff spot')
    # for x in range(1,teams_to_play_off+1):
    #     print(str(x)+'\t'+str(round((avg_wins[x-1])/simulations,2)))

    delta = datetime.datetime.now() - begin

    # print('')
    # print('Histrogram of wins required for final playoff spot')
    # for x in range(1,len(last_playoff_wins)+1):
    #     print(str(x)+'\t'+str(round((last_playoff_wins[x-1])/(simulations*1.0)*100,3))+'\t'+str(round((first_playoff_miss[x-1])/(simulations*1.0)*100,3)))

    print('\n{0:,}'.format(simulations) + " Simulations ran in " + str(delta))

    return projections

def gen_expected_standings(power_rankings):
    """ By comparing the current power score of a team to their remaining opponents, project wins and losses for the rest of the year"""
    expected_wins = [] # empty list to be filled with # of expected wins for each team
    sos = [] # strength of schedule list

    for team in league.teams:
        # Get the team's Power Score by matching the team name in the DataFrame
        team_power_score = float(power_rankings.loc[power_rankings['Team'] == team.team_name, 'Power Score'].values[0])

        # Empty list to populate with the teams probability of winning each remaining matchup
        win_prob_schedule = []

        # Initialize strength of schedule to 0, will be average power score of remaining opponents
        team_sos = 0

        # Calculate team's win prob for each opp on schedule, only for remaining games
        for week_num, opp in enumerate(team.schedule, start=1):
            if week_num > week:  # Skip previous games, only calculate for remaining weeks
                # Get the current power score of the opponent
                opp_power_score = float(
                    power_rankings.loc[power_rankings['Team'] == opp.team_name, 'Power Score'].values[0])

                # Compute the probability of the team winning as the team's share of the added power scores of both teams
                win_prob = team_power_score / (opp_power_score + team_power_score)

                # Add opponent's power score to SOS value
                team_sos += opp_power_score

                # Add probability to win this matchup to win prob schedule list
                win_prob_schedule.append(win_prob)

        # Calculate a team's expected wins as the sum of its win probabilities and current wins
        team_expected_wins = round(sum(win_prob_schedule) + team.wins, 2)

        # Calculate expected losses as weeks remaining - win_prob + current losses
        team_expected_losses = round(15 - week - sum(win_prob_schedule) + team.losses, 2)

        # Average SOS by remaining games
        team_sos = team_sos / (15 - week)

        # Append team name and expected wins to league wide expected wins list
        expected_wins.append([team.team_name, team_expected_wins, team_expected_losses])

        # Append team name and remaining sos to league wide sos list
        sos.append([team.team_name, team_sos])

    # Convert expected wins and sos to Dataframes to join together
    expected_wins_df = pd.DataFrame(expected_wins, columns=['Team','Projected Wins','Projected Losses'])
    sos_df = pd.DataFrame(sos, columns=['Team', 'sos']).round()

    # if it is week 9 or greater, join sos with expected wins
    if week > 9:
        expected_wins_df = expected_wins_df.merge(sos_df, on='Team')

    # Sort by projected wins
    expected_wins_df = expected_wins_df.sort_values(by=['Projected Wins'], ascending=False)


    # Set index to start at 1
    expected_wins_df = expected_wins_df.set_axis(range(1, len(expected_wins_df) + 1))

    return expected_wins_df

def gen_ai_summary():
    print("\n\tRetrieving and processing matchups...")

    # Create dataframe of fantasy pros names to generate player urls for news
    names = pd.read_csv('/users/christiangeer/fantasy_sports/football/power_rankings/espn-api-v3/fantasy_pros_names.csv')
    names = pd.DataFrame(names, columns=['Name', 'Team'])

    # load player_values data
    values = pd.read_csv(f'/users/christiangeer/fantasy_sports/football/power_rankings/espn-api-v3/player_values/KTC_values_week{week}.csv')

    urls = []
    # Iterate over each player and their corresponding team
    for player, team in zip(names['Name'], names['Team']):
        # Split the full name into parts and join them with hyphens, all in lowercase
        formatted_name = '-'.join(player.split()).lower()

        # Create the URL with f-string
        url = f"https://www.fantasypros.com/nfl/players/{formatted_name}.php"

        # Append the player name, team, and URL to the urls list
        urls.append([player, team, url])

    # Now urls contains the player, team, and formatted URL

    # Convert urls list to DataFrame
    urls = pd.DataFrame(urls, columns=['Name','url', 'Team'])

    # merge urls df with names df, retain name urls to maintain code below
    urls = pd.merge(urls, names, how='inner', on='Name')

    urls.drop('Team_y', axis=1, inplace=True)
    urls.rename(columns={'Team_x': 'url', 'url': 'Team'}, inplace=True)

    # Fuzzy merge 'values' DataFrame with 'urls' to get player values
    values = fuzzy_merge(
        values[['Player Name', 'Pos', 'Value', 'NFL_Team']],
        urls[['Name', 'url', 'Team']],
        'Player Name',
        'Name',
        'NFL_Team',
        'Team',
        threshold=90
    )

    # Fill 'Name' NaN with 'Player Name'
    values['Name'].fillna(values['Player Name'], inplace=True)

    # drop unnecessary columns
    values.drop(['Best Match', 'Match Score', 'Team', 'Pos', 'Name'], axis=1, inplace=True)

    if week > 1:
        # get player values from last week and merge with current values and urls
        prev_values = pd.read_csv(f'/users/christiangeer/fantasy_sports/football/power_rankings/espn-api-v3/player_values/KTC_values_week{week-1}.csv')

        # convert to df
        prev_values = pd.DataFrame(prev_values, columns=['Player Name', 'Pos', 'Value', 'NFL_Team'])

        # Fuzzy merge 'values' DataFrame with 'urls' to get player values
        values = fuzzy_merge(
            prev_values[['Player Name', 'Pos', 'Value', 'NFL_Team']],
            values[['Player Name', 'Value', 'NFL_Team', 'url']],
            'Player Name',
            'Player Name',
            'NFL_Team',
            'NFL_Team',
            threshold=90
        )

        # drop unnecessary columns from merge and rename value_x and value_y to be clear
        values.drop(['Player Name_y', 'Pos', 'Best Match', 'Match Score','NFL_Team_x', 'NFL_Team_y'], axis=1, inplace=True)
        values.rename(columns={'Player Name_x':'Player Name', 'Value_x':'prev_value', 'Value_y':'value'}, inplace=True)

        # calculate change in values from last week to this  week
        values['change_value'] = values['value'] - values['prev_value']

        # drop raw value data
        values.drop(['value', 'prev_value'], axis=1, inplace=True)

        # Create a dictionary mapping player names to their values
        name_to_value = dict(zip(values['Player Name'], values['change_value']))

        # Create a dictionary mapping player names to their URLs
        name_to_url = dict(zip(values['Player Name'], values['url']))
    else:
        # Create a dictionary mapping player names to their values
        name_to_value = dict(zip(values['Player Name'], values['Value']))

        # Create a dictionary mapping player names to their URLs
        name_to_url = dict(zip(urls['Player Name'], urls['url']))

    # Retrieve all matchups for the given week
    matchups = league.box_scores(week=week)

    # Extract box score data
    box_scores_data = []

    for matchup in matchups:
        matchup_data = {
            "home_team": matchup.home_team.team_name,
            "home_score": matchup.home_score,
            "home_projected": matchup.home_projected,
            "away_team": matchup.away_team.team_name,
            "away_score": matchup.away_score,
            "away_projected": matchup.away_projected,
            "home_players": [
                {
                    "player_name": player.name,
                    "slot_position": player.slot_position,
                    "position": player.position,
                    "points": player.points,
                    "projected_points": player.projected_points,
                    "url": name_to_url.get(player.name, "URL not found"),  # Lookup the URL from the dictionary
                    "value_change": name_to_value.get(player.name, "Value not found")  # Lookup the value from the dictionary
                } for player in matchup.home_lineup
            ],
            "away_players": [
                {
                    "player_name": player.name,
                    "position": player.position,
                    "slot_position": player.slot_position,
                    "points": player.points,
                    "projected_points": player.projected_points,
                    "url": name_to_url.get(player.name, "URL not found"),  # Lookup the URL from the dictionary
                    "value_change": name_to_value.get(player.name, "Value not found")  # Lookup the value from the dictionary
                } for player in matchup.away_lineup
            ]
        }
        box_scores_data.append(matchup_data)

    # Convert to JSON format
    box_scores_json = json.dumps(box_scores_data, indent=4)

    print("\n\tGenerating summary with LLM...")

    # Sample JSON data (replace with your actual JSON data)
    # json_data = box_scores_json

    # Setting up OpenAI model
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.4, openai_api_key=api_key)

    # For repeated testing calls
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.25, openai_api_key=api_key)


    # Define the prompt template for generating a newspaper-like summary
    prompt_template = PromptTemplate(
        input_variables=["box_scores_json"],
        template="""
        Write a newspaper-style summary of the fantasy football matchups based on the following JSON data:

        {box_scores_json}
        
        Important things to know:
        - If a players 'slot_position' is equal to 'BE' or 'IR', then that player was on the team's bench. Otherwise, they were in the starting lineup
            
        The summary should include:
        - The names of the teams
        - Which team won the matchup and if it was close.
        - If a players 'value_change' is greater than  100, or less than -100, browse 'url' for that player. The 'url' web page will contain recent news about that player to provide insights about him. Include those insights in the recap.
        - Performance comparison of a team's players with the same 'position'. Call out when a player on the bench scored more points than a player with the same 'position' in the starting lineup.
        
        The summary can also include:
        - The projected scores for each team
        - Players in the starting lineup that greatly over or under performed their projected points

        Write in a formal, engaging newspaper tone.
        """
    )

    # Initialize the LLMChain with the Llama model and prompt template
    llm_chain = RunnableSequence(
        prompt_template | llm
    )

    # Generate the newspaper-like summary
    result = llm_chain.invoke(input=box_scores_json)

    # return the result
    return result.content

# Generate Power Rankings
print('\nGenerating Power Rankings...')

rankings, team_values = gen_power_rankings(week)

if week > 1:
    prev_rankings, prev_team_values = gen_power_rankings(week-1)
    print('\nGenerating Last Week\'s Power Rankings...')

    print('\n\tCalculating weekly change...')
    weekly_change_rankings = weekly_change(rankings, prev_rankings)

# Generate Expected Standings
print('\nGenerating expected standings...')
expected_standings = gen_expected_standings(rankings)

# Generate Playoff Probability (if week 5 or later) and append to expected standings
if week >= 5:
    print('\nGenerating playoff probabilities...')
    playoff_prob = gen_playoff_prob()

# Generate Luck Index
print('\nGenerating Luck Index...')
bar_luck_index = progressbar.ProgressBar(max_value=len(teams))

season_luck_index = []
luck_index_value = 0
for i, team in enumerate(teams):
    team_name = team.team_name
    for luck_week in range(1, week+1):
        luck_index_value += luck_index.get_weekly_luck_index(league, team, luck_week)

    # append team's season long luck index to the list
    season_luck_index.append([team.team_name, luck_index_value])

    # reset luck index value
    luck_index_value = 0

    # Update the progress bar
    bar_luck_index.update(i + 1)

# convert season long luck index list to pandas dataframe, sort by 'Luck Index', and set index to start at 1
season_luck_index = pd.DataFrame(season_luck_index, columns=['Team','Luck Index'])
season_luck_index['Luck Index'] = round(season_luck_index['Luck Index'],2 )
season_luck_index.sort_values(by='Luck Index', ascending=False, inplace=True, ignore_index=True)
season_luck_index = season_luck_index.set_axis(range(1, len(season_luck_index)+1))


# Generate AI Summary
print('\n\nGenerating AI Summary...')
summary = gen_ai_summary()

# Print everything
print('\nWriting to markdown file...')
# open text file
filepath = f"/Users/christiangeer/Fantasy_Sports/football/power_rankings/jtown-dynasty/content/blog/Week{week}{year}PowerRankings.md"
sys.stdout = open(filepath, "w")

# for the markdown files in blog
print("---")
print(f"title: Week {week} {year} Report")
print(f"date: {datetime.now().date()}")
print(f"image: /images/{year}week{week}.jpg")
print("draft: false")
print("---")

print("<!-- excerpt -->")

print("\n# POWER RANKINGS\n")
# Value un-informed
print(table(weekly_change_rankings, headers='keys', tablefmt='pipe', colalign=('center', 'left','center','center','center','center'))) # have to manually center all play % because its not a number

print('\n', summary)

if week >= 5:
    print("\n## Current Playoff Probabilities")
    print(table(playoff_prob, headers='keys', tablefmt='pipe', numalign='center'))

print(f"\n## Projected Standings (as of week {week})")
print(table(expected_standings, headers='keys', tablefmt='pipe', numalign='center'))

print("\n## LUCK INDEX")
print(table(season_luck_index, headers='keys', tablefmt='pipe', numalign='center'))

# Close file and restore standard output
sys.stdout.close()
sys.stdout = sys.__stdout__

print('\nDone!\n')
