import nfl_data_py as nfl
import numpy as np
import torch
import random
from torch.utils.data import Dataset


class GameDataSet(Dataset):
    """
    Data Set class for NFL game samples
    """
    def __init__(self, statistics, outcomes):
        """
        Initializes a GameDataSet from general team stats and game outcomes
        :param statistics: dictionary of each teams statistics for offense and defense
        :param outcomes: outcomes of each game
        """
        self.games: [([float], int)] = []
        for outcome in outcomes:
            self.games.append((
                torch.tensor(
                    # Concatenate offense and defense stats for each time
                    list(statistics[outcome[1]]["o"].values()) +
                    list(statistics[outcome[1]]["d"].values()) +
                    list(statistics[outcome[2]]["o"].values()) +
                    list(statistics[outcome[2]]["d"].values()),
                    dtype=torch.float
                ),
                outcome[0]
            ))

    def __len__(self):
        return len(self.games)

    def __getitem__(self, item):
        tup = self.games[item]
        data = tup[0]
        label = tup[1]
        return data, label

def evaluate(plays, exps):
    """
    Evaluate the provided statistic expressions across all given plays
    :param plays: Relevant plays
    :param exps: Statistic expressions to be evaluated
    :return: A dictionary of each expression name to its value
    """

    # Count number of drives and plays represented in the given sample
    drives = len(set([play["fixed_drive"] for play in plays]))
    count = len(plays)

    # Compile all required expression keys
    keys = set([key for exp in exps for key in exp.keys()])

    # Sum required statistics over all plays
    totals = {
        key: sum([
            play[key] for play in plays if
            isinstance(play[key], np.float64) and not np.isnan(play[key])
        ])
        for key in keys
    }

    # Evaluate each expression
    return {exp.__str__(): exp.eval(totals, drives, count) for exp in exps}


def calculate_team(plays, exps, role):
    """
    Calculates statistics for both the team as a whole and for every matchup
    :param plays: Relevant plays to calculate over
    :param exps: Statistic expressions to calculate
    :param role: Role of the team, either offense "o" or defense "d"
    :return: A tuple containing general team stats and matchup stats
    """

    # Determine
    comp = "defteam" if role == "o" else "posteam"
    opponents = set([play[comp] for play in plays])

    # Evaluate general team statistics over all plays
    general = evaluate(plays, exps)
    """
    # Evaluate stats against each opponent
    opp_results = dict()
    for opp in opponents:
        relevant = evaluate([play for play in plays if play[comp] == opp], exps)
        opp_results[opp] = relevant
        """

    return general


def stats(exps):
    """
    Downloads NFL base statistics, extracts useful information, and calculates statistics for each team and matchup
    :param exps: Statistic expressions to extract and calculate
    :return: Dictionary mapping team names and offense/defense roles to general and matchup-specific stats and game outcomes
    """

    # Download play-by-play data for relevant years
    years = [2023]
    df = nfl.import_pbp_data(years, downcast=False)
    teams = dict()
    games = nfl.import_schedules(years)
    results = []
    for result, home, away in zip(games["result"], games["home_team"], games["away_team"]):
        results.append([0 if result > 0 else 1, home, away])
    random.shuffle(results)

    # Sort plays into each team's offense and defense depending on possession
    for i in range(df.shape[0]):
        play = df.loc[i]
        posteam = play["posteam"]
        defteam = play["defteam"]
        if None in [posteam, defteam]:
            continue
        for t in [posteam, defteam]:
            if t not in teams:
                teams[t] = {"o": [], "d": []}
        teams[posteam]["o"].append(play)
        teams[defteam]["d"].append(play)

    # Compile general and matchup-specific team stats
    team_results = dict()
    for team, plays in teams.items():
        offense_total = calculate_team(plays["o"], exps, "o")
        defense_total = calculate_team(plays["d"], exps, "d")
        team_results[team] = {"o": offense_total, "d": defense_total}

    return (team_results, results)

def create_datasets(data):
    """
    Build shuffled training, validation, and testing data sets from game data
    :param data:
    :return:
    """
    team_results, results = data
    random.shuffle(results)
    # Split data 60/20/20
    training = GameDataSet(team_results, results[0:int(0.6 * len(results))])
    validation = GameDataSet(team_results, results[int(0.6 * len(results)):int(0.8 * len(results))])
    testing = GameDataSet(team_results, results[int(0.8 * len(results)):])

    return training, validation, testing
