"""Automated tests for the Smart Waste AI project core algorithms."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from waste_management.food_waste_prediction import make_synthetic_data, train_model, predict
from waste_management.waste_rules import classify_item, needs_collection
from practical_03_bfs_dfs import bfs, dfs, GRID
from practical_04_astar import astar, GRID as AGRID, START, TARGET
from practical_05_minimax import legal_moves, is_terminal, evaluate, START_BOARD


class TestFoodWastePrediction(unittest.TestCase):
    def test_data_shape(self):
        df = make_synthetic_data()
        self.assertEqual(df.shape[1], 5)
        self.assertGreater(len(df), 0)

    def test_model_trains(self):
        df = make_synthetic_data(60)
        model, X_test, y_test, preds, metrics = train_model(df)
        self.assertIn("MAE", metrics)
        self.assertIn("R2", metrics)
        self.assertEqual(len(preds), len(y_test))

    def test_predict_positive(self):
        df = make_synthetic_data(60)
        model, *_ = train_model(df)
        result = predict(model, 100, 2, 0, 20.0)
        self.assertGreater(result, 0)


class TestWasteRules(unittest.TestCase):
    def test_food_wet(self):
        r = classify_item("food scraps", {})
        self.assertEqual(r["category_key"], "wet")

    def test_soiled_paper_residual(self):
        r = classify_item("paper", {"soiled": True})
        self.assertEqual(r["category_key"], "residual")

    def test_clean_paper_dry(self):
        r = classify_item("paper", {"soiled": False})
        self.assertEqual(r["category_key"], "dry")

    def test_unknown_residual(self):
        r = classify_item("mystery object", {})
        self.assertEqual(r["category_key"], "residual")

    def test_collection(self):
        self.assertTrue(needs_collection("any", 85.0))
        self.assertFalse(needs_collection("any", 50.0))


class TestSearch(unittest.TestCase):
    def test_bfs_reaches(self):
        path, steps = bfs(GRID)
        self.assertIsNotNone(path)
        self.assertIsNotNone(steps)

    def test_dfs_reaches(self):
        path, steps = dfs(GRID)
        self.assertIsNotNone(path)

    def test_bfs_shortest(self):
        path, steps = bfs(GRID)
        # BFS on unweighted grid should find a path of length <= DFS
        _, dfs_steps = dfs(GRID)
        self.assertLessEqual(steps, dfs_steps)

    def test_unreachable(self):
        grid = [["S", "#"], ["#", "T"]]
        path, steps = bfs(grid)
        self.assertIsNone(path)
        path2, _ = astar([[0, 1], [1, 0]], (0, 0), (1, 1))
        self.assertIsNone(path2)

    def test_astar_cost(self):
        path, cost = astar(AGRID, START, TARGET)
        self.assertIsNotNone(path)
        self.assertGreater(cost, 0)


class TestMinimax(unittest.TestCase):
    def test_legal_moves_start(self):
        moves = legal_moves(START_BOARD, "X")
        self.assertGreater(len(moves), 0)

    def test_terminal_full(self):
        full = [["X"] * 5 for _ in range(5)]
        self.assertTrue(is_terminal(full))

    def test_evaluate(self):
        score = evaluate(START_BOARD, "X")
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
