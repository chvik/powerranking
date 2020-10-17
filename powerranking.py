import re
import sys
from collections import defaultdict
import numpy as np

def read_data(file_name):
    teams = set()
    matches = []
    by_opponents = defaultdict(list)
    line_pattern = re.compile(r'^\s*(\S.*?)\s*-\s*(\S.*?)\s*(\d+)\s*-\s*(\d+)\s*$')
    with open(file_name) as f:
        for line in f:
            match = line_pattern.match(line)
            if not match:
                print("ignored line:", line.rstrip(), file=sys.stderr)
            home_team, away_team, home_goals, away_goals = match.group(1, 2, 3, 4)
            teams.add(home_team)
            teams.add(away_team)
            matches.append((home_team, away_team, home_goals, away_goals))
            by_opponents[(home_team, away_team)].append((home_goals, away_goals))
    return {
        "teams": sorted(teams),
        "matches": matches,
        "by_opponents": by_opponents
    }

def calculate_normalized_scores(league):
    base_scores = defaultdict(int)
    n_games = defaultdict(int)
    for match in league["matches"]:
        home_team, away_team, home_goals, away_goals = match
        n_games[home_team] += 1
        n_games[away_team] += 1
        if home_goals > away_goals:
            base_scores[home_team] += 3
        elif away_goals > home_goals:
            base_scores[away_team] += 3
        else:
            base_scores[home_team] += 1
            base_scores[away_team] += 1
    
    team_quotients = {}
    for team in league["teams"]:
        team_quotients[team] = base_scores[team] / n_games[team]
    avg_quotient = sum(team_quotients.values()) / len(team_quotients)

    normalized_scores = []
    for team in league["teams"]:
        normalized_scores.append(team_quotients[team] - avg_quotient)
    return np.array(normalized_scores)


def get_laplacian_matrix(league):
    matrix = []
    for i, team_i in enumerate(league["teams"]):
        row = []
        for j, team_j in enumerate(league["teams"]):
            if i == j:
                row.append(None)
            else:
                n_games_ij = len(league["by_opponents"][(team_i, team_j)]) + len(league["by_opponents"][(team_j, team_i)])
                row.append(-n_games_ij)
        row[i] = -sum(row[:i] + row[i+1:])
        matrix.append(row)
    return np.array(matrix)


def least_square(laplacian, scores, r, i):
    if i == 0:
        return (1/r) * scores
    elif i > 0:
        q_prev = least_square(laplacian, scores, r, i-1)
        return q_prev + 1/r * (1/r * (r * np.identity(laplacian.shape[0]) - laplacian))**i @ scores
