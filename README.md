# espn-api-v3

This project aims to make ESPN Fantasy Football statistics easily available. 
With the introduction of version 3 of the ESPN's API, this structure creates leagues, teams, and player classes that allow for advanced data analytics and the potential for many new features to be added.

I am new to the Git interface, but any recommendations and pull requests are welcome.

This project was inspired and based off of [rbarton65/espnff](https://github.com/rbarton65/espnff)

Additional help/ideas were received from [cwendt94/ff-espn-api](https://github.com/cwendt94/ff-espn-api)

## Table of Contents  
* Fetching league
  * [Fetch public leagues](##fetchpublicleagues)
  * [Fetch private leagues](##fetchprivateleagues)
* Viewing league information
  * [View league information](##viewleagueinformation)
  * [View team information](##viewteaminformation)
  * [View player information](##viewplayerinformation)
  * [View stats for a specific week](##viewstatsforaspecificweek)
* Analytic Methods
  * [Power Rankings](##powerrankings)
  * [Luck Index](##luckindex)
  * [Projected Standings](##projectedstandings)


<a name="fetchpublicleagues"></a>
## Fetch public leagues
In the main.py file, type:
```python
>>> league_id = 1234
>>> year = 2019
>>> league = League(league_id, year)
```

<a name="fetchprivateleagues"></a>
## Fetch private leagues
By typing your script into the main.py file:
```python
>>> league_id = 1234
>>> year = 2019
>>> user = yourusername@website.com
>>> pass = yourPassword
>>> league = League(league_id, year, user, pass)
Fetching league...
League authenticated!
Gathering team information...
Gathering matchup data...
Gathering roster settings information...
Current Week: 3
Building teams...
Building schedule...
	Building week 1/12...
	Building week 2/12...
	Building week 3/12...
	Building week 4/12...
	Building week 5/12...
	Building week 6/12...
	Building week 7/12...
	Building week 8/12...
	Building week 9/12...
	Building week 10/12...
	Building week 11/12...
	Building week 12/12...
League successfully built!
League(La Lega di Cugino, 2019)
```

<a name="viewleagueinformation"></a>
## View league information
```python
>>> league.year
2019
>>> league.currentWeek
2
>>> league.regSeasonWeeks
12
>>> self.numTeams
8
>>> league.teamNames
{1: ['John Smith', 'T.Y. Very Much'], 2: ['Jane Doe', 'Home Sweet Mahomes'], ... teamId: [owner name, team name]}
>>> league.teams
{1: Team(T.Y. Very Much), 2: Team(Home Sweet Mahomes), ... teamId: Team(Team n Name)}
```

<a name="viewteaminformation"></a>
## View team information
```python
>>> team = league.teams[1]
Team(T.Y. Very Much')
>>> team.owner
'John Smith'
>>> team.teamName
'T.Y. Very Much'
>>> team.abbrev
'TYVM'
>>> team.wins
2
>>> team.losses
0
>>> team.schedule
{1: Team(Home Sweet Mahomes), 2: Team(Can you Diggs this?), .... weekNum: Team(opponentName) }
>>> team.scores
{1: 163.7, 2: 124.2, ... weekNum: score }
```
Under team.rosters, each value in the dictionary contains a list of player objects that relate to the team's roster for the given week.
```python
>>> team.rosters
{1: [Player(Ezekiel Elliot), Player(Kyler Murray), .....], 2: [Player(Todd Gurley), Player(Kyler Murray) .... ] 
```

<a name="viewplayerinformation"></a>
## View player information
```python
>>> player = team.rosters[1][0]
>>> player.name
'Ezekiel Elliot'
>>> player.id
3051392
>>> player.eligibleSlots	# position slot ids that the player can be placed in
[2, 3, 23, 7, 20, 21]
>>> player.positionId		# position slot id that the user used him in
2
>>> player.isStarting
True
```

<a name="viewstatsforaspecificweek"></a>
## View stats for a specific week
The two main purposes for this package is to be able to quickly and seamlessly view stats for a team or league that ESPN doesn't readily compute.
Using the 'printWeeklyStats' method, you can view a weekly report for a certain week.
```python
>>> team.printWeeklyStats(1)
----------------------------
John Smith Week 1
----------------------------
Week Score: 149.9
Best Possible Lineup: 156.42
Opponent Score: 116.5
Weekly Finish: 3
Best Trio: 74.32
Number of Injuries: 0
Starting QB pts: 30.72
Avg. Starting RB pts: 23.6
Avg. Starting WR pts: 9.85
Starting TE pts: 5.7
Starting Flex pts: 19.6
Starting DST pts: 10.0
Starting K pts: 17.0
Total Bench pts: 71.12
----------------------------
>>> league.printWeeklyStats(1)
 Week 1
---------------------  ----------------
Most Points Scored:    Marco
Least Points Scored:   Ellie
Best Possible Lineup:  Desi
Best Trio:             Desi
Worst Trio:            Vincent
---------------------  ----------------
Best QBs:              Nikki
Best RBs:              Desi
Best WRs:              Nikki
Best TEs:              Isabella
Best Flex:             Julia
Best DST:              Marc
Best K:                Ellie
Best Bench:            Ellie
---------------------  ----------------
Worst QBs:             Desi
Worst RBs:             Ellie
Worst WRs:             Julia
Worst TEs:             Vincent
Worst Flex:            Nikki
Worst DST:             Vincent
Worst K:               Marc
Worst Bench:           Gabriel
```

<a name="powerrankings"></a>
## Power Rankings
This package has its own formula for calculating power rankings each week. 
The computation takes in a team's performance over the entire season (with more weight on the recent weeks), while also accounting for luck.
The power rankings for a given week can be viewed using the `printPowerRankings` method.
```python
>>> league.printPowerRankings(1)
 Week  1 
 Power Index                      Team  Owner
-----------------------------  ------  ----------------
The Adams Family               101.52  Marc Chirico
T.Y. Very Much                 101.24  Desi Pilla
Sony with a Chance              93.02  Isabella Chirico
Good Ole   Christian Boys       79.57  Gabriel S
Home Sweet Mahomes              76.30  Nikki  Pilla
Any Tom, Dick,  Harry Will Do   70.96  Vincent Chirico
The Kamara adds 10 pounds       65.41  Julia Selleck
Can you Diggs this?             64.38  Ellie Knecht
```

<a name="luckindex"></a>
## Luck Index
This package has its own formula for calculating how "lucky" a team has been over the course of a season. Each week, every team is assigned a luck score
based on how they did relative to the rest of the league and the result of their weekly matchup. Teams that performed poorly but still won are assigned 
a higher score, while teams that did well but still lost are assigned lower scores. The other determinant in a team's weekly luck score is how well they performed
relative to their average performance, as well as how their opponent performed relative to their average score. Team's who scored exceptionally higher than they
normally do will have a higher luck score, and vice versa. Likewise, team's who face opponents that over-acheive relative to their typical performance will have
a lower (or more 'unlucky') score. Over the course of the season, the luck scores are totaled and the luck index is compiled. The luck index can be viewed using
 the `printLuckIndex` method.
```python
>>> league.printLuckIndex(2)
Through Week 2
 Team                         Luck Index  Owner
-------------------------  ------------  ----------------
Can you Diggs this?                4.29  Ellie Knecht
Sony with a Chance                 2.14  Isabella Chirico
T.Y. Very Much                     0.71  Desi Pilla
The Adams Family                   0     Marc Chirico
Good Ole   Christian Boys          0     Gabriel S
Home Sweet Mahomes                -1.43  Nikki  Pilla
The Kamara adds 10 pounds         -2.14  Julia Selleck
All Tom No Jerry                  -3.57  Vincent Chirico
```
 
## Projected Standings
Using the power rankings calculated by this package, projections for the final standings can be calculated. The `printExpectedStandings` method can be called to 
view the expected standings based on the power rankings through a certain week. The current standings are found, and results of the following matchups are predicted.
For example, if week 2 has just concluded, the most up-to-date projections can be viewed as follows:

```python
>>> league.printExpectedStandings(2)
Week 2 
 Team                         Wins    Losses    Ties  Owner
-------------------------  ------  --------  ------  ----------------
T.Y. Very Much                 12         0       0  Desi Pilla
Sony with a Chance             11         1       0  Isabella Chirico
Home Sweet Mahomes              8         4       0  Nikki  Pilla
The Adams Family                6         6       0  Marc Chirico
Good Ole   Christian Boys       5         7       0  Gabriel S
All Tom No Jerry                3         9       0  Vincent Chirico
Can you Diggs this?             3         9       0  Ellie Knecht
The Kamara adds 10 pounds       0        12       0  Julia Selleck

*These standings do not account for tiesbreakers
```
