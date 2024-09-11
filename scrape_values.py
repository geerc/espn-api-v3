from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from progressbar import ProgressBar
from tqdm import tqdm
from tabulate import tabulate as table
import requests
import csv
from tqdm import tqdm

root = '/Users/christiangeer/Fantasy_Sports/football/power_rankings/player_values/espn-api-v3/values/week'


def player_values(week):
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
    with tqdm(total=total_players, desc="Scraping Player Data") as pbar:
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

    # Write to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Player Name', 'Pos', 'Value'])
        writer.writeheader()  # Write header
        writer.writerows(players)  # Write rows from player data

    print(f"Data successfully written to {csv_file_path}\n")

    # Convert to Pandas DF
    players_df = pd.DataFrame(players, columns=['Player Name','Pos','Value'])

    return players_df

player_values = player_values(1)
print(player_values.head())