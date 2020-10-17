from unittest import TestCase
import numpy as np
from powerranking import (
    read_data,
    calculate_normalized_scores,
    get_laplacian_matrix,
    least_square,
)


class PowerRankingTest(TestCase):
    def test_read_data(self):
        league = read_data("test-data.txt")
        self.assertListEqual(
            league["teams"], ["Barcelona", "Bayern", "Juventus", "Liverpool"]
        )
        self.assertListEqual(
            league["matches"],
            [
                ("Bayern", "Juventus", "2", "1"),
                ("Barcelona", "Liverpool", "2", "0"),
                ("Bayern", "Barcelona", "1", "0"),
            ],
        )
        self.assertDictEqual(
            league["by_opponents"],
            {
                ("Bayern", "Juventus"): [("2", "1")],
                ("Barcelona", "Liverpool"): [("2", "0")],
                ("Bayern", "Barcelona"): [("1", "0")],
            },
        )

    def test_calculate_normalized_scores(self):
        league = read_data("test-data.txt")
        normalized_scores = calculate_normalized_scores(league)
        self.assertTrue(
            np.array_equal(normalized_scores, np.array([0.375, 1.875, -1.125, -1.125]))
        )

    def test_laplacian_matrix(self):
        league = read_data("test-data.txt")
        matrix = get_laplacian_matrix(league)
        self.assertListEqual(
            matrix.tolist(),
            [[2, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 1, 0], [-1, 0, 0, 1]],
        )

    def test_least_square(self):
        league = read_data("nb1-202021.txt")
        normalized_scores = calculate_normalized_scores(league)
        laplacian = get_laplacian_matrix(league)
        r = np.amax(np.diagonal(laplacian))
        print(league["teams"])
        print(self.format_result(league["teams"], normalized_scores))
        for k in [0, 1, 2, 3, 4, 10]:
            print(self.format_result(league["teams"], least_square(laplacian, normalized_scores, r, k)))

    def format_result(self, teams, ratings):
        team_rating_pairs = zip(teams, ratings)
        sorted_pairs = sorted(team_rating_pairs, key=lambda pair: pair[1], reverse=True)
        print(sum(ratings)/len(ratings))
        return sorted_pairs