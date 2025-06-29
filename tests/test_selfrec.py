<<<<<<< HEAD


import os
import sys
import pytest
from unittest.mock import MagicMock, patch
=======
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, mock_open
>>>>>>> f12e1c0 (Comprehensive test suite implementation for main functional modules)

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from SELFRec import SELFRec
<<<<<<< HEAD
from util.conf import ModelConf
from data.loader import FileIO

@pytest.fixture
def setup_selfrec_test_data(tmp_path):
    """
    Fixture to set up dummy data files and a ModelConf object for SELFRec testing.
    """
    # Create dummy data files
    train_file = tmp_path / "train.txt"
    test_file = tmp_path / "test.txt"
    valid_file = tmp_path / "valid.txt"
    social_file = tmp_path / "social.txt"

    train_content = "u1 i1 1.0\nu2 i2 1.0\n"
    test_content = "u3 i3 1.0\n"
    valid_content = "u4 i4 1.0\n"
    social_content = "u1 u2 0.5\nu3 u4 0.8\n"

    train_file.write_text(train_content)
    test_file.write_text(test_content)
    valid_file.write_text(valid_content)
    social_file.write_text(social_content)

    # Create a dummy ModelConf
    conf_file_path = tmp_path / "test_model.conf"
    conf_content = f"""
training.set={train_file}
test.set={test_file}
valid.set={valid_file}
model.type=graph
model.name=LightGCN
social.data={social_file}
"""
    conf_file_path.write_text(conf_content)
    config = ModelConf(str(conf_file_path))

    return config, train_file, test_file, valid_file, social_file

def test_selfrec_init(setup_selfrec_test_data):
    """
    Test the __init__ method of SELFRec to ensure data is loaded correctly.
    """
    config, train_file, test_file, valid_file, social_file = setup_selfrec_test_data

    # Mock FileIO methods to control data loading and avoid actual file I/O during init
    with patch('data.loader.FileIO.load_data_set') as mock_load_data_set, \
         patch('data.loader.FileIO.load_social_data') as mock_load_social_data:

        # Define return values for mocked methods
        mock_load_data_set.side_effect = [
            [['u1', 'i1', 1.0], ['u2', 'i2', 1.0]], # for training.set
            [['u3', 'i3', 1.0]], # for test.set
            [['u4', 'i4', 1.0]]  # for valid.set
        ]
        mock_load_social_data.return_value = [['u1', 'u2', 0.5], ['u3', 'u4', 0.8]]

        rec = SELFRec(config)

        # Assertions for data loading
        mock_load_data_set.assert_any_call(str(train_file), 'graph')
        mock_load_data_set.assert_any_call(str(test_file), 'graph')
        mock_load_data_set.assert_any_call(str(valid_file), 'graph')
        mock_load_social_data.assert_called_once_with(str(social_file))

        assert rec.training_data == [['u1', 'i1', 1.0], ['u2', 'i2', 1.0]]
        assert rec.test_data == [['u3', 'i3', 1.0]]
        assert rec.valid_data == [['u4', 'i4', 1.0]]
        assert rec.kwargs['social.data'] == [['u1', 'u2', 0.5], ['u3', 'u4', 0.8]]
        assert rec.config == config

def test_selfrec_init_no_social_data(tmp_path):
    """
    Test the __init__ method of SELFRec when no social data is configured.
    """
    # Create dummy data files
    train_file = tmp_path / "train_no_social.txt"
    test_file = tmp_path / "test_no_social.txt"
    valid_file = tmp_path / "valid_no_social.txt"

    train_content = "u1 i1 1.0\n"
    test_content = "u2 i2 1.0\n"
    valid_content = "u3 i3 1.0\n"

    train_file.write_text(train_content)
    test_file.write_text(test_content)
    valid_file.write_text(valid_content)

    # Create a dummy ModelConf without social.data
    conf_file_path = tmp_path / "test_model_no_social.conf"
    conf_content = f"""
training.set={train_file}
test.set={test_file}
valid.set={valid_file}
model.type=graph
model.name=LightGCN
"""
    conf_file_path.write_text(conf_content)
    config = ModelConf(str(conf_file_path))

    with patch('data.loader.FileIO.load_data_set') as mock_load_data_set, \
         patch('data.loader.FileIO.load_social_data') as mock_load_social_data:

        mock_load_data_set.side_effect = [
            [['u1', 'i1', 1.0]],
            [['u2', 'i2', 1.0]],
            [['u3', 'i3', 1.0]]
        ]
        mock_load_social_data.return_value = [] # Should not be called, but good to have a default

        rec = SELFRec(config)

        mock_load_data_set.assert_any_call(str(train_file), 'graph')
        mock_load_data_set.assert_any_call(str(test_file), 'graph')
        mock_load_data_set.assert_any_call(str(valid_file), 'graph')
        mock_load_social_data.assert_not_called() # Ensure social data loading is skipped

        assert 'social.data' not in rec.kwargs
        assert rec.config == config

=======

class TestSELFRec:
    """Test class for SELFRec functionality."""

    def create_mock_config(self):
        """Create a mock configuration for testing."""
        from util.conf import ModelConf
        config = MagicMock(spec=ModelConf)
        config.__getitem__.side_effect = lambda key: {
            'training.set': 'data/train.txt',
            'test.set': 'data/test.txt', 
            'valid.set': 'data/valid.txt',
            'model.type': 'graph',
            'model.name': 'TestModel'
        }[key]
        config.contain.return_value = False  # Default: no social data
        return config

    @patch('SELFRec.FileIO.load_data_set')
    def test_selfrec_initialization_basic(self, mock_load_data):
        """Test basic SELFRec initialization."""
        config = self.create_mock_config()
        
        # Mock FileIO.load_data_set to return sample data
        training_data = [['user1', 'item1', 1.0]]
        test_data = [['user1', 'item2', 2.0]]
        valid_data = [['user1', 'item3', 3.0]]
        
        mock_load_data.side_effect = [training_data, test_data, valid_data]
        
        with patch('builtins.print'):  # Suppress print output
            selfrec = SELFRec(config)
        
        # Check that data was loaded correctly
        assert selfrec.training_data == training_data
        assert selfrec.test_data == test_data
        assert selfrec.valid_data == valid_data
        assert selfrec.config == config
        
        # Check FileIO.load_data_set was called correctly
        assert mock_load_data.call_count == 3
        mock_load_data.assert_any_call('data/train.txt', 'graph')
        mock_load_data.assert_any_call('data/test.txt', 'graph')
        mock_load_data.assert_any_call('data/valid.txt', 'graph')

    @patch('SELFRec.FileIO.load_data_set')
    @patch('SELFRec.FileIO.load_social_data')
    def test_selfrec_initialization_with_social_data(self, mock_load_social, mock_load_data):
        """Test SELFRec initialization with social data."""
        config = self.create_mock_config()
        config.__getitem__.side_effect = lambda key: {
            'training.set': 'data/train.txt',
            'test.set': 'data/test.txt', 
            'valid.set': 'data/valid.txt',
            'model.type': 'graph',
            'model.name': 'TestModel',
            'social.data': 'data/social.txt'
        }[key]
        config.contain.return_value = True  # Has social data
        
        # Mock data loading
        training_data = [['user1', 'item1', 1.0]]
        test_data = [['user1', 'item2', 2.0]]
        valid_data = [['user1', 'item3', 3.0]]
        social_data = [['user1', 'user2', 1.0]]
        
        mock_load_data.side_effect = [training_data, test_data, valid_data]
        mock_load_social.return_value = social_data
        
        with patch('builtins.print'):  # Suppress print output
            selfrec = SELFRec(config)
        
        # Check that social data was loaded
        assert 'social.data' in selfrec.kwargs
        assert selfrec.kwargs['social.data'] == social_data
        mock_load_social.assert_called_once_with('data/social.txt')

    @patch('SELFRec.FileIO.load_data_set')
    def test_selfrec_initialization_without_social_data(self, mock_load_data):
        """Test SELFRec initialization without social data."""
        config = self.create_mock_config()
        # No social.data key in config
        
        # Mock data loading
        training_data = [['user1', 'item1', 1.0]]
        test_data = [['user1', 'item2', 2.0]]
        valid_data = [['user1', 'item3', 3.0]]
        
        mock_load_data.side_effect = [training_data, test_data, valid_data]
        
        with patch('builtins.print'):  # Suppress print output
            selfrec = SELFRec(config)
        
        # Check that kwargs is empty
        assert selfrec.kwargs == {}

    @patch('SELFRec.FileIO.load_data_set')
    def test_selfrec_initialization_attributes(self, mock_load_data):
        """Test that SELFRec initializes all required attributes."""
        config = self.create_mock_config()
        
        mock_load_data.return_value = []
        
        with patch('builtins.print'):  # Suppress print output
            selfrec = SELFRec(config)
        
        # Check all attributes are initialized
        assert hasattr(selfrec, 'social_data')
        assert hasattr(selfrec, 'feature_data')
        assert hasattr(selfrec, 'config')
        assert hasattr(selfrec, 'training_data')
        assert hasattr(selfrec, 'test_data')
        assert hasattr(selfrec, 'valid_data')
        assert hasattr(selfrec, 'kwargs')
        
        # Check initial values
        assert selfrec.social_data == []
        assert selfrec.feature_data == []

    @patch('SELFRec.FileIO.load_data_set')
    def test_execute_method_imports_and_runs_model(self, mock_load_data):
        """Test execute method imports and runs the correct model."""
        config = self.create_mock_config()
        
        mock_load_data.return_value = []
        
        with patch('builtins.print'):  # Suppress print output
            selfrec = SELFRec(config)
        
        # Mock the model class and its execute method
        mock_model_instance = MagicMock()
        mock_model_class = MagicMock(return_value=mock_model_instance)
        
        # Patch exec and eval to control model import and instantiation
        with patch('builtins.exec') as mock_exec, \
             patch('builtins.eval', return_value=mock_model_instance) as mock_eval:
            
            selfrec.execute()
            
            # Check that exec was called to import the model
            mock_exec.assert_called_once()
            exec_call_args = mock_exec.call_args[0][0]
            assert 'from model.graph.TestModel import TestModel' in exec_call_args
            
            # Check that eval was called to instantiate and execute the model
            mock_eval.assert_called_once()
            eval_call_args = mock_eval.call_args[0][0]
            expected_eval_str = 'TestModel(self.config,self.training_data,self.test_data,self.valid_data,**self.kwargs)'
            assert eval_call_args == expected_eval_str
            
            # Check that model's execute method was called
            mock_model_instance.execute.assert_called_once()

    @patch('SELFRec.FileIO.load_data_set')
    def test_execute_with_different_model_types(self, mock_load_data):
        """Test execute method with different model types."""
        mock_load_data.return_value = []
        
        # Test with sequential model type
        from util.conf import ModelConf
        config = MagicMock(spec=ModelConf)
        config.__getitem__.side_effect = lambda key: {
            'training.set': 'data/train.txt',
            'test.set': 'data/test.txt',
            'valid.set': 'data/valid.txt',
            'model.type': 'sequential',
            'model.name': 'SeqModel'
        }[key]
        config.contain.return_value = False
        
        with patch('builtins.print'):  # Suppress print output
            selfrec = SELFRec(config)
        
        mock_model_instance = MagicMock()
        
        with patch('builtins.exec') as mock_exec, \
             patch('builtins.eval', return_value=mock_model_instance):
            
            selfrec.execute()
            
            # Check that correct import string was generated
            exec_call_args = mock_exec.call_args[0][0]
            assert 'from model.sequential.SeqModel import SeqModel' in exec_call_args

    @patch('SELFRec.FileIO.load_data_set')
    def test_execute_with_social_data_kwargs(self, mock_load_data):
        """Test execute method passes social data through kwargs."""
        config = self.create_mock_config()
        config.__getitem__.side_effect = lambda key: {
            'training.set': 'data/train.txt',
            'test.set': 'data/test.txt', 
            'valid.set': 'data/valid.txt',
            'model.type': 'graph',
            'model.name': 'TestModel',
            'social.data': 'data/social.txt'
        }[key]
        config.contain.return_value = True  # Has social data
        
        mock_load_data.return_value = []
        
        with patch('builtins.print'), \
             patch('SELFRec.FileIO.load_social_data', return_value=[['u1', 'u2', 1.0]]):
            selfrec = SELFRec(config)
        
        mock_model_instance = MagicMock()
        
        with patch('builtins.exec'), \
             patch('builtins.eval', return_value=mock_model_instance) as mock_eval:
            
            selfrec.execute()
            
            # Check that kwargs containing social data is passed to model
            eval_call_args = mock_eval.call_args[0][0]
            assert '**self.kwargs' in eval_call_args

    @patch('SELFRec.FileIO.load_data_set')
    def test_print_output_during_initialization(self, mock_load_data):
        """Test that initialization prints the expected message."""
        config = self.create_mock_config()
        mock_load_data.return_value = []
        
        with patch('builtins.print') as mock_print:
            SELFRec(config)
            
            # Check that the expected message was printed
            mock_print.assert_called_with('Reading data and preprocessing...')

    @patch('SELFRec.FileIO.load_data_set')
    def test_config_contain_method_usage(self, mock_load_data):
        """Test that config.contain method is used correctly for social data."""
        # Mock config object with contain method
        from util.conf import ModelConf
        config = MagicMock(spec=ModelConf)
        config.__getitem__.side_effect = lambda key: {
            'training.set': 'data/train.txt',
            'test.set': 'data/test.txt',
            'valid.set': 'data/valid.txt',
            'model.type': 'graph',
            'social.data': 'data/social.txt'
        }[key]
        
        # Test when social.data exists
        config.contain.return_value = True
        
        mock_load_data.return_value = []
        
        with patch('builtins.print'), \
             patch('SELFRec.FileIO.load_social_data', return_value=[]) as mock_social:
            
            selfrec = SELFRec(config)
            
            # Check that contain was called with 'social.data'
            config.contain.assert_called_with('social.data')
            # Check that social data was loaded
            mock_social.assert_called_once()

    @patch('SELFRec.FileIO.load_data_set')
    def test_config_contain_method_false(self, mock_load_data):
        """Test behavior when config.contain returns False for social data."""
        # Mock config object with contain method
        from util.conf import ModelConf
        config = MagicMock(spec=ModelConf)
        config.__getitem__.side_effect = lambda key: {
            'training.set': 'data/train.txt',
            'test.set': 'data/test.txt',
            'valid.set': 'data/valid.txt',
            'model.type': 'graph'
        }[key]
        
        # Test when social.data does not exist
        config.contain.return_value = False
        
        mock_load_data.return_value = []
        
        with patch('builtins.print'), \
             patch('SELFRec.FileIO.load_social_data') as mock_social:
            
            selfrec = SELFRec(config)
            
            # Check that contain was called with 'social.data'
            config.contain.assert_called_with('social.data')
            # Check that social data was NOT loaded
            mock_social.assert_not_called()
            # Check that kwargs is empty
            assert selfrec.kwargs == {}
>>>>>>> f12e1c0 (Comprehensive test suite implementation for main functional modules)
