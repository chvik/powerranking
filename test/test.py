from unittest import TestCase
import numpy as np

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from powerranking import (
    read_data,
    calculate_absolute_scores,
    calculate_normalized_scores,
    get_laplacian_matrix,
    least_square,
)


class PowerRankingTest(TestCase):
    def test_read_data(self):
        league = read_data(parent / "test-data.txt")
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

    def test_calculate_absolute_scores(self):
        league = read_data(parent / "test-data.txt")
        (absolute_scores, n_games) = calculate_absolute_scores(league)
        self.assertDictEqual(
            absolute_scores, {
                "Barcelona": 3, 
                "Bayern": 6, 
            }
        )
        self.assertDictEqual(
            n_games, {
                "Barcelona": 2,
                "Bayern": 2,
                "Liverpool": 1,
                "Juventus": 1,
            }
        )

    def test_calculate_normalized_scores(self):
        league = read_data(parent / "test-data.txt")
        normalized_scores = calculate_normalized_scores(league)
        self.assertTrue(
            np.array_equal(normalized_scores, np.array([0.375, 1.875, -1.125, -1.125]))
        )

    def test_laplacian_matrix(self):
        league = read_data(parent / "test-data.txt")
        matrix = get_laplacian_matrix(league)
        self.assertListEqual(
            matrix.tolist(),
            [[2, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 1, 0], [-1, 0, 0, 1]],
        )
