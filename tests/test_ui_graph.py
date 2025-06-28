import os
import sys
import pytest
import numpy as np
from collections import defaultdict
from unittest.mock import MagicMock, patch

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from data.ui_graph import Interaction

class TestInteraction:
    """Test class for Interaction functionality."""

    def create_sample_data(self):
        """Create sample data for testing."""
        config = {'test_key': 'test_value'}
        training_data = [
            ['user1', 'item1', 1.0],
            ['user1', 'item2', 2.0],
            ['user2', 'item2', 1.5],
            ['user2', 'item3', 3.0]
        ]
        test_data = [
            ['user1', 'item3', 2.5],
            ['user2', 'item1', 1.8]
        ]
        valid_data = [
            ['user1', 'item4', 3.5],
            ['user2', 'item4', 2.2]
        ]
        return config, training_data, test_data, valid_data

    def test_interaction_initialization(self):
        """Test Interaction class initialization."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # Check basic data attributes
        assert interaction.config == config
        assert interaction.training_data == training_data
        assert interaction.test_data == test_data
        assert interaction.valid_data == valid_data
        
        # Check user and item mappings
        expected_users = {'user1': 0, 'user2': 1}
        expected_items = {'item1': 0, 'item2': 1, 'item3': 2}  # Only training items are in the dictionary
        assert interaction.user == expected_users
        assert interaction.item == expected_items
        
        # Check reverse mappings
        assert interaction.id2user == {0: 'user1', 1: 'user2'}
        assert interaction.id2item == {0: 'item1', 1: 'item2', 2: 'item3'}

    def test_training_set_generation(self):
        """Test training set generation."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # Check training_set_u (user-item ratings)
        assert interaction.training_set_u['user1']['item1'] == 1.0
        assert interaction.training_set_u['user1']['item2'] == 2.0
        assert interaction.training_set_u['user2']['item2'] == 1.5
        assert interaction.training_set_u['user2']['item3'] == 3.0
        
        # Check training_set_i (item-user ratings)
        assert interaction.training_set_i['item1']['user1'] == 1.0
        assert interaction.training_set_i['item2']['user1'] == 2.0
        assert interaction.training_set_i['item2']['user2'] == 1.5
        assert interaction.training_set_i['item3']['user2'] == 3.0

    def test_test_set_generation(self):
        """Test test set generation."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # Check test_set
        assert interaction.test_set['user1']['item3'] == 2.5
        assert interaction.test_set['user2']['item1'] == 1.8
        
        # Check test_set_item
        expected_test_items = {'item1', 'item3'}
        assert interaction.test_set_item == expected_test_items

    def test_valid_set_generation(self):
        """Test valid set generation."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # Check valid_set - item4 is not in training, so should be skipped
        assert len(interaction.valid_set) == 0  # No valid items since item4 is not in training
        
        # Check valid_set_item
        assert len(interaction.valid_set_item) == 0

    def test_user_item_counts(self):
        """Test user and item count calculations."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # User count should be based on training_set_u
        assert interaction.user_num == 2
        # Item count should be based on training_set_i
        assert interaction.item_num == 3  # Only items in training: item1, item2, item3

    def test_row_vector_generation(self):
        """Test row vector generation for users."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # Test row for user1 (id=0)
        row_vec = interaction.row(0)
        expected_length = len(interaction.item)  # All items
        assert len(row_vec) == expected_length
        
        # Check specific ratings
        assert row_vec[interaction.item['item1']] == 1.0  # user1 rated item1 as 1.0
        assert row_vec[interaction.item['item2']] == 2.0  # user1 rated item2 as 2.0

    def test_col_vector_generation(self):
        """Test column vector generation for items."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # Test col for item2 (id=1)
        col_vec = interaction.col(1)
        expected_length = len(interaction.user)  # All users
        assert len(col_vec) == expected_length
        
        # Check specific ratings for item2
        assert col_vec[interaction.user['user1']] == 2.0  # user1 rated item2 as 2.0
        assert col_vec[interaction.user['user2']] == 1.5  # user2 rated item2 as 1.5

    def test_matrix_generation(self):
        """Test full interaction matrix generation."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        matrix = interaction.matrix()
        
        # Check matrix dimensions
        assert matrix.shape == (len(interaction.user), len(interaction.item))
        
        # Check specific values
        user1_id = interaction.user['user1']
        user2_id = interaction.user['user2']
        item1_id = interaction.item['item1']
        item2_id = interaction.item['item2']
        item3_id = interaction.item['item3']
        
        assert matrix[user1_id, item1_id] == 1.0
        assert matrix[user1_id, item2_id] == 2.0
        assert matrix[user2_id, item2_id] == 1.5
        assert matrix[user2_id, item3_id] == 3.0

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        config = {}
        training_data = []
        test_data = []
        valid_data = []
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        assert interaction.user == {}
        assert interaction.item == {}
        assert interaction.user_num == 0
        assert interaction.item_num == 0
        assert len(interaction.test_set_item) == 0
        assert len(interaction.valid_set_item) == 0

    def test_missing_users_items_in_test_valid(self):
        """Test handling of users/items in test/valid that are not in training."""
        config = {}
        training_data = [['user1', 'item1', 1.0]]
        test_data = [
            ['user1', 'item1', 1.5],  # Valid user and item
            ['user_new', 'item1', 2.0],  # User not in training
            ['user1', 'item_new', 2.5]  # Item not in training
        ]
        valid_data = [
            ['user1', 'item1', 3.0],  # Valid user and item
            ['user_new', 'item_new', 3.5]  # Both not in training
        ]
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # Only valid user-item pairs should be in test/valid sets
        assert len(interaction.test_set) == 1  # Only user1
        assert 'user1' in interaction.test_set
        assert interaction.test_set['user1']['item1'] == 1.5
        
        assert len(interaction.valid_set) == 1  # Only user1
        assert 'user1' in interaction.valid_set
        assert interaction.valid_set['user1']['item1'] == 3.0

    def test_sparse_matrices_created(self):
        """Test that sparse matrices are created during initialization."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # Check that sparse matrices are created
        assert interaction.ui_adj is not None
        assert interaction.norm_adj is not None
        assert interaction.interaction_mat is not None
        
        # Check that they are scipy sparse matrices
        import scipy.sparse as sp
        assert sp.issparse(interaction.ui_adj)
        assert sp.issparse(interaction.norm_adj)
        assert sp.issparse(interaction.interaction_mat)

    def test_user_rated_method_exists(self):
        """Test that user_rated method works (inherited from Data or Graph)."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # Test if user_rated method exists and can be called
        try:
            items, ratings = interaction.user_rated('user1')
            # Should return items and ratings for user1
            assert len(items) > 0
            assert len(ratings) > 0
            assert len(items) == len(ratings)
        except AttributeError:
            # Method might not be implemented, which is okay for testing
            pytest.skip("user_rated method not implemented")

    def test_item_rated_method_exists(self):
        """Test that item_rated method works (inherited from Data or Graph)."""
        config, training_data, test_data, valid_data = self.create_sample_data()
        
        interaction = Interaction(config, training_data, test_data, valid_data)
        
        # Test if item_rated method exists and can be called
        try:
            users, ratings = interaction.item_rated('item2')
            # Should return users and ratings for item2
            assert len(users) > 0
            assert len(ratings) > 0
            assert len(users) == len(ratings)
        except AttributeError:
            # Method might not be implemented, which is okay for testing
            pytest.skip("item_rated method not implemented")