import os
import sys
import pytest

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from data.data import Data

class TestData:
    """Test class for Data functionality."""

    def test_data_initialization(self):
        """Test Data class initialization."""
        # Mock configuration
        config = {'test_key': 'test_value'}
        training_data = [['user1', 'item1', 1.0], ['user2', 'item2', 2.0]]
        test_data = [['user1', 'item3', 3.0]]
        valid_data = [['user2', 'item4', 4.0]]
        
        data = Data(config, training_data, test_data, valid_data)
        
        assert data.config == config
        assert data.training_data == training_data
        assert data.test_data == test_data
        assert data.valid_data == valid_data

    def test_data_empty_initialization(self):
        """Test Data class initialization with empty data."""
        config = {}
        training_data = []
        test_data = []
        valid_data = []
        
        data = Data(config, training_data, test_data, valid_data)
        
        assert data.config == config
        assert data.training_data == []
        assert data.test_data == []
        assert data.valid_data == []

    def test_data_none_initialization(self):
        """Test Data class initialization with None values."""
        config = None
        training_data = None
        test_data = None
        valid_data = None
        
        data = Data(config, training_data, test_data, valid_data)
        
        assert data.config is None
        assert data.training_data is None
        assert data.test_data is None
        assert data.valid_data is None

    def test_data_different_types(self):
        """Test Data class with different data types."""
        config = {'model': 'test_model', 'epochs': 10}
        
        # Training data as list of lists
        training_data = [['u1', 'i1', 1.0], ['u2', 'i2', 2.0]]
        
        # Test data as different format
        test_data = {'u1': {'i3': 3.0}, 'u2': {'i4': 4.0}}
        
        # Valid data as strings
        valid_data = "u1 i5 5.0\nu2 i6 6.0"
        
        data = Data(config, training_data, test_data, valid_data)
        
        assert data.config == config
        assert data.training_data == training_data
        assert data.test_data == test_data
        assert data.valid_data == valid_data