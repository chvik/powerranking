#!/usr/bin/env python3

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


def calculate_absolute_scores(league):
    absolute_scores = defaultdict(int)
    n_games = defaultdict(int)
    for match in league["matches"]:
        home_team, away_team, home_goals, away_goals = match
        n_games[home_team] += 1
        n_games[away_team] += 1
        if home_goals > away_goals:
            absolute_scores[home_team] += 3
        elif away_goals > home_goals:
            absolute_scores[away_team] += 3
        else:
            absolute_scores[home_team] += 1
            absolute_scores[away_team] += 1
    return (absolute_scores, n_games)


def calculate_normalized_scores(league):
    absolute_scores, n_games = calculate_absolute_scores(league)
    
    team_quotients = {}
    for team in league["teams"]:
        team_quotients[team] = absolute_scores[team] / n_games[team]
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


def powerranking_report(args):
    if len(args) == 0:
        print("Missing argument: results file name", file=sys.stderr)
        exit(1)

    league = read_data(args[0])
    absolute_scores, n_games = calculate_absolute_scores(league)
    normalized_scores = calculate_normalized_scores(league)
    laplacian = get_laplacian_matrix(league)
    r = np.amax(np.diagonal(laplacian))
    
    print("Standing by absolute score (no tiebreaker)")
    print_formatted_absolute_scores(league["teams"], absolute_scores, n_games)
    print()

    print("Standing by normalized scores")
    print_formatted_result(league["teams"], normalized_scores)
    print()

    iteration = 10
    print(f"Standing by least square algorithm (after {iteration} iterations)")
    print_formatted_result(league["teams"], least_square(laplacian, normalized_scores, r, iteration))
    print()


def print_formatted_absolute_scores(teams, absolute_score, n_games):
    team_score_n_games = zip(teams, [absolute_score[team] for team in teams], [n_games[team] for team in teams])
    sorted_triplets = sorted(team_score_n_games, key=lambda triplet: triplet[1], reverse=True)
    for i, (team, score, n) in enumerate(sorted_triplets):
        if i > 0 and score == sorted_triplets[i-1][1]:
            # tie
            print(f"  {team} {score} {n}")
        else:
            print(f"{i+1} {team} {score} {n}")


def print_formatted_result(teams, ratings):
    team_rating_pairs = zip(teams, ratings)
    sorted_pairs = sorted(team_rating_pairs, key=lambda pair: pair[1], reverse=True)
    for i, (team, score) in enumerate(sorted_pairs):
        if i > 0 and score == sorted_pairs[i-1][1]:
            # tie
            print(f"  {team} {score}")
        else:
            print(f"{i+1} {team} {score}") 


if __name__ == "__main__":
    powerranking_report(sys.argv[1:])