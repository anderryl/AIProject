import nfl_data_py as nfl
from expression import Exp
import numpy as np


def evaluate(plays, exps):
    """
    Evaluate the provided statistic expressions across all given plays
    :param plays: Relevant plays
    :param exps: Statistic expressions to be evaluated
    :return: A dictionary of each expression name to its value
    """
    drives = len(set([play["fixed_drive"] for play in plays]))
    count = len(plays)
    keys = set([key for exp in exps for key in exp.keys()])
    totals = {
        key: sum([
            play[key] for play in plays if
            isinstance(play[key], np.float64) and not np.isnan(play[key])
        ])
        for key in keys
    }
    return {exp.__str__(): exp.eval(totals, drives, count) for exp in exps}


def calculate_team(plays, exps, role):
    """
    Calculates statistics for both the team as a whole and for every matchup
    :param plays: Relevant plays to calculate over
    :param exps: Statistic expressions to calculate
    :param role: Role of the team, either offense "o" or defense "d"
    :return: A tuple containing general team stats and matchup stats
    """
    comp = "defteam" if role == "o" else "posteam"
    opponents = set([play[comp] for play in plays])
    stats = evaluate(plays, exps)
    opp_results = dict()
    for opp in opponents:
        relevant = evaluate([play for play in plays if play[comp] == opp], exps)
        opp_results[opp] = relevant
    return stats, opp_results


def stats(exps):
    """
    Downloads NFL base statistics, extracts useful information, and calculates statistics for each team and matchup
    :param exps: Statistic expressions to extract and calculate
    :return: Dictionary mapping team names and offense/defense roles to general and matchup-specific stats
    """

    # Download play-by-play data for relevant years
    years = [2023]
    df = nfl.import_pbp_data(years, downcast=False)
    teams = dict()

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
        offense_total, offense_matchups = calculate_team(plays["o"], exps, "o")
        defense_total, defense_matchups = calculate_team(plays["d"], exps, "d")
        team_results[team] = {"o": (offense_total, offense_matchups), "d": (defense_total, defense_matchups)}

    return team_results

# Statistic expressions to be calculated
expressions = [
    Exp("complete_pass") / Exp("pass") // "completion_rate",
    Exp("rushing_yards") / Exp("rush") // "rushing_avg",
    Exp("receiving_yards") / Exp("pass") // "passing_avg",
    Exp("penalty") / Exp("plays") // "penalty_avg",
    Exp("sack") / Exp("plays") // "sack_rate",
    Exp("fumble") / Exp("plays") // "fumble_rate",
    Exp("interception") / Exp("pass") // "intercept_rate",
    Exp("yards_gained") / Exp("drives") // "drive_yards",
    Exp("first_down") / Exp("drives") // "drive_firsts",
]

# Print final results
print(stats(expressions))
