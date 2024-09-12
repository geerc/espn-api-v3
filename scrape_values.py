'''
TODO:
1. Scrape Dynasty values for missing players, join with redraft values
2. Replace print statements with logging statements
3. Send log as email?
'''

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate as table
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import os
from espn_api.football import League
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Get login credentials for leagues
league_id = os.getenv('league_id')
swid = os.getenv('swid')
espn_s2 = os.getenv('espn_s2')
api_key= os.getenv('OPEN_AI_KEY')

# Define year
year = datetime.now().year

league = League(league_id, year, espn_s2, swid)
print(league, "\n")

# Get NFL week
week = league.nfl_week - 1

def scrape_redraft_values(week):
    # URL of the rankings page
    base_url = "https://keeptradecut.com/fantasy-rankings?page={}&filters=QB|WR|RB|TE|DST|PK&format=1"

    # CSV file path
    csv_file_path = f'/users/christiangeer/fantasy_sports/football/power_rankings/espn-api-v3/player_values/KTC_values_week{week}.csv'

    # Find the table or section containing the player data
    players = []

    # First, we estimate the total number of player rows to set the progress bar
    total_players = 0
    for page in range(8):
        url = base_url.format(page)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        player_rows = soup.find_all('div', class_='onePlayer')
        total_players += len(player_rows)

    # Initialize a single tqdm progress bar for all pages
    with tqdm(total=total_players, desc="Scraping Redraft Values") as pbar:
        # Loop over pages 0 to 7
        for page in range(8):
            url = base_url.format(page)

            # Send a request to fetch the webpage for each page
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all the player rows
            player_rows = soup.find_all('div', class_='onePlayer')

            # Iterate through the relevant sections containing player name, position, and value
            for row in player_rows:
                player_info = row.find('div', class_='player-name')

                # Extract player name from <a> tag
                player_name = player_info.find('a').get_text(strip=True)

                # Extract player position and value
                player_pos = row.find('p', class_='position').get_text(strip=True)
                player_value = row.find('div', class_='value').get_text(strip=True)

                # Append to players list
                players.append({
                    'Player Name': player_name,
                    'Pos': player_pos,
                    'Value': player_value
                })

                # Update progress bar
                pbar.update(1)

    # Convert to Pandas DF
    players_df = pd.DataFrame(players, columns=['Player Name','Pos','Value'])

    return players_df

def scrape_dynasty_values(week):
    # URL of the rankings page
    dynasty_url = "https://keeptradecut.com/dynasty-rankings?page={}&filters=QB|WR|RB|TE&format=1"

    # Find the table or section containing the player data
    dynasty_players = []

    # First, we estimate the total number of player rows to set the progress bar
    total_dynasty_players = 0
    for page in range(10):
        url = dynasty_url.format(page)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        player_rows = soup.find_all('div', class_='onePlayer')
        total_dynasty_players += len(player_rows)

    # Initialize a single tqdm progress bar for all pages
    with tqdm(total=total_dynasty_players, desc="Scraping Dynasty Values") as pbar:
        # Loop over pages 0 to 9
        for page in range(10):
            url = dynasty_url.format(page)

            # Send a request to fetch the webpage for each page
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all the player rows
            player_rows = soup.find_all('div', class_='onePlayer')

            # Iterate through the relevant sections containing player name, position, and value
            for row in player_rows:
                player_info = row.find('div', class_='player-name')

                # Extract player name from <a> tag
                player_name = player_info.find('a').get_text(strip=True)

                # Extract player position and value
                player_pos = row.find('p', class_='position').get_text(strip=True)
                player_value = row.find('div', class_='value').get_text(strip=True)

                # Append to players list
                dynasty_players.append({
                    'Player Name': player_name,
                    'Pos': player_pos,
                    'Value': player_value
                })

                # Update progress bar
                pbar.update(1)

        # Convert dynasty values to Pandas Dataframe
        dynasty_values = pd.DataFrame(dynasty_players, columns=['Player Name','Pos','Value'])

        # Convert 'Value' to numeric, forcing errors to NaN
        dynasty_values['Value'] = pd.to_numeric(dynasty_values['Value'].str.replace(',', '').str.replace('$', ''), errors='coerce')

        return dynasty_values

def merge_vales(redraft, dynasty, week):
    print('Merging...')

    # Perform a left join to get all rows from redraft and the matching rows from dynasty
    merged_values = pd.merge(redraft, dynasty, on='Player Name', how='left', suffixes=('_redraft', '_dynasty'))

    # Find rows in dynasty that don't have a match in redraft
    unmatched_dynasty = dynasty[~dynasty['Player Name'].isin(redraft['Player Name'])]

    # Concatenate the unmatched rows with the merged DataFrame
    final_df = pd.concat([merged_values, unmatched_dynasty], ignore_index=True)

    # Fill NaN values in Value_redraft and Pos_redraft with Value and Pos from unmatched rows
    final_df['Value'] = final_df['Value_redraft'].fillna(final_df['Value'])
    final_df['Pos'] = final_df['Pos_redraft'].fillna(final_df['Pos'])

    # Select only the required columns
    final_df = final_df[['Player Name', 'Pos', 'Value']]

    return final_df

# Scrape Dynasty and Redraft values
dynasty_values = scrape_dynasty_values(week)

# Adjust dyansty_values to more closely reflect potential redraft values
dynasty_values['Value'] = dynasty_values['Value']*.8

redraft_values =  scrape_redraft_values(week)

# Merge values
merged_values = merge_vales(redraft_values, dynasty_values, week)
print('Merged Values:\n', merged_values)

# Write to a CSV file
merged_values[['Player Name', 'Pos', 'Value']].to_csv(path_or_buf=f'/users/christiangeer/fantasy_sports/football/power_rankings/espn-api-v3/player_values/KTC_values_week{week}.csv')


