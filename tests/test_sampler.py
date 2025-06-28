import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from util.sampler import (
    next_batch_pairwise, 
    next_batch_pairwise_fl, 
    next_batch_pointwise
)

class MockData:
    """Mock data class for testing sampling functions."""
    
    def __init__(self):
        self.training_data = [
            ['user1', 'item1', 1.0],
            ['user1', 'item2', 1.0], 
            ['user2', 'item3', 1.0],
            ['user2', 'item4', 1.0]
        ]
        self.user = {'user1': 0, 'user2': 1}
        self.item = {'item1': 0, 'item2': 1, 'item3': 2, 'item4': 3, 'item5': 4}
        self.item_num = len(self.item)
        self.id2item = {0: 'item1', 1: 'item2', 2: 'item3', 3: 'item4', 4: 'item5'}
        self.training_set_u = {
            'user1': {'item1': 1.0, 'item2': 1.0},
            'user2': {'item3': 1.0, 'item4': 1.0}
        }


class TestSamplingFunctions:
    """Test class for sampling functions."""

    def test_next_batch_pairwise_basic(self):
        """Test basic pairwise batch generation."""
        data = MockData()
        batch_size = 2
        
        batches = list(next_batch_pairwise(data, batch_size, n_negs=1))
        
        assert len(batches) > 0
        for u_idx, i_idx, j_idx in batches:
            assert len(u_idx) == len(i_idx) == len(j_idx)
            assert all(isinstance(u, int) for u in u_idx)
            assert all(isinstance(i, int) for i in i_idx)
            assert all(isinstance(j, int) for j in j_idx)

    def test_next_batch_pairwise_single_batch(self):
        """Test pairwise batch when data fits in single batch."""
        data = MockData()
        batch_size = 10  # Larger than training data
        
        batches = list(next_batch_pairwise(data, batch_size, n_negs=1))
        
        assert len(batches) == 1
        u_idx, i_idx, j_idx = batches[0]
        assert len(u_idx) == len(data.training_data)

    def test_next_batch_pairwise_multiple_negatives(self):
        """Test pairwise batch with multiple negative samples."""
        data = MockData()
        batch_size = 2
        n_negs = 3
        
        batches = list(next_batch_pairwise(data, batch_size, n_negs=n_negs))
        
        for u_idx, i_idx, j_idx in batches:
            # Each positive sample should have n_negs negative samples
            assert len(j_idx) == len(i_idx) * n_negs

    def test_next_batch_pairwise_fl_basic(self):
        """Test federated learning pairwise batch generation."""
        data = MockData()
        batch_size = 2
        select_user_list = ['user1', 'user2']
        
        batches = list(next_batch_pairwise_fl(data, batch_size, select_user_list, n_negs=1))
        
        assert len(batches) == len(select_user_list)
        for u_idx, i_idx, j_idx in batches:
            assert len(u_idx) == len(i_idx) == len(j_idx)

    def test_next_batch_pairwise_fl_single_user(self):
        """Test federated learning batch with single user."""
        data = MockData()
        batch_size = 5
        select_user_list = ['user1']
        
        batches = list(next_batch_pairwise_fl(data, batch_size, select_user_list, n_negs=1))
        
        assert len(batches) == 1
        u_idx, i_idx, j_idx = batches[0]
        # user1 has 2 interactions
        assert len(u_idx) == 2

    def test_next_batch_pairwise_fl_empty_user_list(self):
        """Test federated learning batch with empty user list."""
        data = MockData()
        batch_size = 2
        select_user_list = []
        
        batches = list(next_batch_pairwise_fl(data, batch_size, select_user_list, n_negs=1))
        
        assert len(batches) == 0

    def test_next_batch_pointwise_basic(self):
        """Test basic pointwise batch generation."""
        data = MockData()
        batch_size = 2
        
        batches = list(next_batch_pointwise(data, batch_size))
        
        assert len(batches) > 0
        for u_idx, i_idx, y in batches:
            assert len(u_idx) == len(i_idx) == len(y)
            # Each positive sample generates 4 negative samples (as per implementation)
            expected_length = (batch_size * 5)  # 1 positive + 4 negatives per sample
            assert len(u_idx) == expected_length or len(u_idx) < expected_length  # Could be smaller for last batch

    def test_next_batch_pointwise_labels(self):
        """Test pointwise batch labels are correct."""
        data = MockData()
        batch_size = 1
        
        batches = list(next_batch_pointwise(data, batch_size))
        
        for u_idx, i_idx, y in batches:
            # Check that labels alternate: 1 positive, then 4 negatives
            for idx in range(0, len(y), 5):
                if idx < len(y):
                    assert y[idx] == 1  # Positive sample
                    for neg_idx in range(idx + 1, min(idx + 5, len(y))):
                        assert y[neg_idx] == 0  # Negative samples

    def test_next_batch_pointwise_single_batch(self):
        """Test pointwise batch when data fits in single batch."""
        data = MockData()
        batch_size = 10  # Larger than training data
        
        batches = list(next_batch_pointwise(data, batch_size))
        
        assert len(batches) == 1

    def test_next_batch_empty_training_data(self):
        """Test batch generation with empty training data."""
        data = MockData()
        data.training_data = []
        batch_size = 2
        
        batches = list(next_batch_pairwise(data, batch_size))
        
        assert len(batches) == 0

    def test_batch_indices_valid_range(self):
        """Test that all generated indices are within valid range."""
        data = MockData()
        batch_size = 2
        
        batches = list(next_batch_pairwise(data, batch_size))
        
        for u_idx, i_idx, j_idx in batches:
            # User indices should be in range [0, num_users)
            assert all(0 <= u < len(data.user) for u in u_idx)
            # Item indices should be in range [0, num_items)
            assert all(0 <= i < len(data.item) for i in i_idx)
            assert all(0 <= j < len(data.item) for j in j_idx)

    @patch('util.sampler.choice')
    def test_negative_sampling_excludes_positive_items(self, mock_choice):
        """Test that negative sampling excludes positive items."""
        data = MockData()
        batch_size = 1
        
        # Mock choice to cycle through values - first positive item, then negative item
        mock_choice.side_effect = ['item1', 'item5'] * 10  # Repeat to handle multiple calls
        
        batches = list(next_batch_pairwise(data, batch_size, n_negs=1))
        
        # The function should call choice at least twice per negative sample
        assert mock_choice.call_count >= 2

    def test_batch_size_consistency(self):
        """Test that batch sizes are consistent."""
        data = MockData()
        batch_size = 2
        
        batches = list(next_batch_pairwise(data, batch_size))
        
        # All batches except possibly the last should have the expected size
        for i, (u_idx, i_idx, j_idx) in enumerate(batches[:-1]):
            assert len(u_idx) == batch_size
        
        # Last batch should have size <= batch_size
        if batches:
            last_u_idx, last_i_idx, last_j_idx = batches[-1]
            assert len(last_u_idx) <= batch_size

    def test_federated_learning_user_specific_batches(self):
        """Test that FL batches are user-specific."""
        data = MockData()
        batch_size = 5
        select_user_list = ['user1']
        
        batches = list(next_batch_pairwise_fl(data, batch_size, select_user_list))
        
        u_idx, i_idx, j_idx = batches[0]
        # All user indices should correspond to user1 (index 0)
        assert all(u == 0 for u in u_idx)

    def test_pointwise_negative_items_not_in_training(self):
        """Test that pointwise negative items are not in user's training set."""
        data = MockData()
        batch_size = 1
        
        # Patch randint to control negative item selection
        with patch('util.sampler.randint') as mock_randint:
            # Cycle through values - first positive item (should be rejected), then negative items
            mock_randint.side_effect = [0, 4] * 20  # 0 maps to item1 (positive), 4 maps to item5 (negative)
            
            batches = list(next_batch_pointwise(data, batch_size))
            
            # Function should call randint multiple times to avoid positive items
            assert mock_randint.call_count >= 6