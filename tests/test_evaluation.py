import os
import sys
import pytest
import math
from unittest.mock import patch

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from util.evaluation import Metric, ranking_evaluation, rating_evaluation

class TestMetric:
    """Test class for Metric functionality."""

    def test_hits_basic(self):
        """Test hits calculation with basic data."""
        origin = {
            'user1': {'item1': 1.0, 'item2': 1.0},
            'user2': {'item3': 1.0, 'item4': 1.0, 'item5': 1.0}
        }
        res = {
            'user1': [('item1', 0.8), ('item3', 0.7)],  # 1 hit
            'user2': [('item3', 0.9), ('item4', 0.8), ('item6', 0.7)]  # 2 hits
        }
        
        hits = Metric.hits(origin, res)
        
        expected = {'user1': 1, 'user2': 2}
        assert hits == expected

    def test_hits_no_matches(self):
        """Test hits calculation with no matches."""
        origin = {
            'user1': {'item1': 1.0, 'item2': 1.0}
        }
        res = {
            'user1': [('item3', 0.8), ('item4', 0.7)]
        }
        
        hits = Metric.hits(origin, res)
        
        expected = {'user1': 0}
        assert hits == expected

    def test_hits_empty_data(self):
        """Test hits calculation with empty data."""
        origin = {}
        res = {}
        
        hits = Metric.hits(origin, res)
        
        assert hits == {}

    def test_hit_ratio_calculation(self):
        """Test hit ratio calculation."""
        origin = {
            'user1': {'item1': 1.0, 'item2': 1.0},  # 2 items
            'user2': {'item3': 1.0, 'item4': 1.0, 'item5': 1.0}  # 3 items, total = 5
        }
        hits = {'user1': 1, 'user2': 2}  # 3 hits out of 5 total = 0.6
        
        hit_ratio = Metric.hit_ratio(origin, hits)
        
        assert hit_ratio == 0.6

    def test_hit_ratio_zero_hits(self):
        """Test hit ratio with zero hits."""
        origin = {
            'user1': {'item1': 1.0, 'item2': 1.0}
        }
        hits = {'user1': 0}
        
        hit_ratio = Metric.hit_ratio(origin, hits)
        
        assert hit_ratio == 0.0

    def test_precision_calculation(self):
        """Test precision calculation."""
        hits = {'user1': 2, 'user2': 1}  # 3 hits total, 2 users
        N = 5  # Top-5 recommendations per user
        
        # Precision = total_hits / (num_users * N) = 3 / (2 * 5) = 0.3
        precision = Metric.precision(hits, N)
        
        assert precision == 0.3

    def test_precision_zero_hits(self):
        """Test precision with zero hits."""
        hits = {'user1': 0, 'user2': 0}
        N = 5
        
        precision = Metric.precision(hits, N)
        
        assert precision == 0.0

    def test_recall_calculation(self):
        """Test recall calculation."""
        hits = {'user1': 2, 'user2': 1}
        origin = {
            'user1': {'item1': 1.0, 'item2': 1.0, 'item3': 1.0},  # 3 items, recall = 2/3
            'user2': {'item4': 1.0, 'item5': 1.0}  # 2 items, recall = 1/2
        }
        
        # Average recall = (2/3 + 1/2) / 2 = (0.6667 + 0.5) / 2 = 0.58335
        recall = Metric.recall(hits, origin)
        
        assert abs(recall - 0.58333) < 0.001

    def test_f1_calculation(self):
        """Test F1 score calculation."""
        prec = 0.3
        recall = 0.4
        
        # F1 = 2 * prec * recall / (prec + recall) = 2 * 0.3 * 0.4 / 0.7 = 0.34286
        f1 = Metric.F1(prec, recall)
        
        assert abs(f1 - 0.34286) < 0.001

    def test_f1_zero_values(self):
        """Test F1 score with zero values."""
        f1_zero_prec = Metric.F1(0.0, 0.5)
        f1_zero_recall = Metric.F1(0.5, 0.0)
        f1_both_zero = Metric.F1(0.0, 0.0)
        
        assert f1_zero_prec == 0
        assert f1_zero_recall == 0
        assert f1_both_zero == 0

    def test_mae_calculation(self):
        """Test MAE (Mean Absolute Error) calculation."""
        res = [
            ('user1', 'item1', 4.0, 3.5),  # error = 0.5
            ('user1', 'item2', 3.0, 3.8),  # error = 0.8
            ('user2', 'item3', 5.0, 4.2),  # error = 0.8
        ]
        
        # MAE = (0.5 + 0.8 + 0.8) / 3 = 0.7
        mae = Metric.MAE(res)
        
        assert abs(mae - 0.7) < 0.001

    def test_mae_empty_result(self):
        """Test MAE with empty result."""
        res = []
        
        mae = Metric.MAE(res)
        
        assert mae == 0

    def test_rmse_calculation(self):
        """Test RMSE (Root Mean Square Error) calculation."""
        res = [
            ('user1', 'item1', 4.0, 3.0),  # error^2 = 1.0
            ('user1', 'item2', 3.0, 4.0),  # error^2 = 1.0
            ('user2', 'item3', 5.0, 3.0),  # error^2 = 4.0
        ]
        
        # RMSE = sqrt((1.0 + 1.0 + 4.0) / 3) = sqrt(2.0) = 1.41421
        rmse = Metric.RMSE(res)
        
        assert abs(rmse - 1.41421) < 0.001

    def test_rmse_empty_result(self):
        """Test RMSE with empty result."""
        res = []
        
        rmse = Metric.RMSE(res)
        
        assert rmse == 0

    def test_ndcg_calculation(self):
        """Test NDCG (Normalized Discounted Cumulative Gain) calculation."""
        origin = {
            'user1': {'item1': 1.0, 'item2': 1.0}
        }
        res = {
            'user1': [('item1', 0.9), ('item3', 0.8), ('item2', 0.7)]  # item1 and item2 are relevant
        }
        N = 3
        
        # DCG = 1/log(2,2) + 0 + 1/log(4,2) = 1 + 0 + 0.5 = 1.5
        # IDCG = 1/log(2,2) + 1/log(3,2) = 1 + 0.63093 = 1.63093
        # NDCG = 1.5 / 1.63093 = 0.91974
        ndcg = Metric.NDCG(origin, res, N)
        
        assert abs(ndcg - 0.91974) < 0.001

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        origin = {
            'user1': {'item1': 1.0, 'item2': 1.0}
        }
        res = {
            'user1': [('item1', 0.9), ('item2', 0.8)]  # Perfect order
        }
        N = 2
        
        # Perfect ranking should give NDCG = 1.0
        ndcg = Metric.NDCG(origin, res, N)
        
        assert ndcg == 1.0


class TestRankingEvaluation:
    """Test class for ranking_evaluation function."""

    def test_ranking_evaluation_basic(self):
        """Test basic ranking evaluation."""
        origin = {
            'user1': {'item1': 1.0, 'item2': 1.0},
            'user2': {'item3': 1.0, 'item4': 1.0}
        }
        res = {
            'user1': [('item1', 0.9), ('item3', 0.8), ('item2', 0.7)],
            'user2': [('item3', 0.9), ('item4', 0.8), ('item5', 0.7)]
        }
        N = [2, 3]
        
        measures = ranking_evaluation(origin, res, N)
        
        assert len(measures) > 0
        assert any('Hit Ratio' in measure for measure in measures)
        assert any('Precision' in measure for measure in measures)
        assert any('Recall' in measure for measure in measures)
        assert any('NDCG' in measure for measure in measures)

    def test_ranking_evaluation_print_for_test(self):
        """Test ranking evaluation with print_for_test=True."""
        origin = {
            'user1': {'item1': 1.0}
        }
        res = {
            'user1': [('item1', 0.9)]
        }
        N = [1]
        
        measures = ranking_evaluation(origin, res, N, print_for_test=True)
        
        assert len(measures) == 1
        assert '\t' in measures[0]  # Tab-separated format

    @patch('builtins.print')
    def test_ranking_evaluation_length_mismatch(self, mock_print):
        """Test ranking evaluation with mismatched lengths."""
        origin = {'user1': {'item1': 1.0}}
        res = {'user1': [('item1', 0.9)], 'user2': [('item2', 0.8)]}  # Extra user
        N = [1]
        
        with pytest.raises(SystemExit):
            ranking_evaluation(origin, res, N)
        
        mock_print.assert_called_with('The Lengths of test set and predicted set do not match!')


class TestRatingEvaluation:
    """Test class for rating_evaluation function."""

    def test_rating_evaluation_basic(self):
        """Test basic rating evaluation."""
        res = [
            ('user1', 'item1', 4.0, 3.5),
            ('user1', 'item2', 3.0, 3.8),
            ('user2', 'item3', 5.0, 4.2),
        ]
        
        measures = rating_evaluation(res)
        
        assert len(measures) == 2
        assert any('MAE' in measure for measure in measures)
        assert any('RMSE' in measure for measure in measures)

    def test_rating_evaluation_empty(self):
        """Test rating evaluation with empty result."""
        res = []
        
        measures = rating_evaluation(res)
        
        assert len(measures) == 2
        assert 'MAE:0' in measures[0]
        assert 'RMSE:0' in measures[1]