import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from base.recommender import Recommender

class TestRecommender:
    """Test class for Recommender functionality."""

    def create_mock_config(self):
        """Create a mock configuration for testing."""
        config = MagicMock()
        config.__getitem__.side_effect = lambda key: {
            'model.name': 'TestModel',
            'item.ranking': 'on -topN 10',
            'embedding.size': '64',
            'num.max.epoch': '100',
            'batch_size': '256',
            'learnRate': '0.001',
            'reg.lambda': '0.01',
            'output.setup': 'on -dir ./output',
            'training.set': '/path/to/train.txt',
            'test.set': '/path/to/test.txt'
        }[key]
        config.config = {
            'model.name': 'TestModel',
            'embedding.size': '64',
            'num.max.epoch': '100'
        }
        config.contain.return_value = False  # Default to no specific parameters
        return config

    @patch('base.recommender.Log')
    @patch('base.recommender.Data')
    @patch('base.recommender.OptionConf')
    def test_recommender_initialization(self, mock_option_conf, mock_data, mock_log):
        """Test Recommender initialization."""
        config = self.create_mock_config()
        training_set = [['user1', 'item1', 1.0]]
        test_set = [['user1', 'item2', 2.0]]
        valid_set = [['user1', 'item3', 3.0]]
        
        # Mock OptionConf instances
        mock_ranking = MagicMock()
        mock_output = MagicMock()
        mock_option_conf.side_effect = [mock_ranking, mock_output]
        
        # Mock Data instance
        mock_data_instance = MagicMock()
        mock_data.return_value = mock_data_instance
        
        # Mock Log instance
        mock_log_instance = MagicMock()
        mock_log.return_value = mock_log_instance
        
        recommender = Recommender(config, training_set, test_set, valid_set)
        
        # Check basic attributes
        assert recommender.config == config
        assert recommender.data == mock_data_instance
        assert recommender.model_name == 'TestModel'
        assert recommender.ranking == mock_ranking
        assert recommender.emb_size == 64
        assert recommender.maxEpoch == 100
        assert recommender.batch_size == 256
        assert recommender.lRate == 0.001
        assert recommender.reg == 0.01
        assert recommender.output == mock_output
        assert recommender.model_log == mock_log_instance
        assert recommender.result == []
        assert recommender.recOutput == []
        
        # Check that Data was initialized correctly
        mock_data.assert_called_once_with(config, training_set, test_set, valid_set)

    @patch('base.recommender.Log')
    @patch('base.recommender.Data')
    @patch('base.recommender.OptionConf')
    def test_recommender_with_kwargs(self, mock_option_conf, mock_data, mock_log):
        """Test Recommender initialization with additional kwargs."""
        config = self.create_mock_config()
        training_set = []
        test_set = []
        valid_set = []
        social_data = [['user1', 'user2', 1.0]]
        
        recommender = Recommender(config, training_set, test_set, valid_set, social_data=social_data)
        
        # The kwargs should be available for subclasses to use
        # (Base class doesn't use them directly)
        assert hasattr(recommender, 'config')

    @patch('base.recommender.Log')
    @patch('base.recommender.Data')
    @patch('base.recommender.OptionConf')
    def test_initializing_log(self, mock_option_conf, mock_data, mock_log):
        """Test initializing_log method."""
        config = self.create_mock_config()
        
        mock_log_instance = MagicMock()
        mock_log.return_value = mock_log_instance
        
        recommender = Recommender(config, [], [], [])
        recommender.initializing_log()
        
        # Check that log.add was called with configuration header
        mock_log_instance.add.assert_any_call('### model configuration ###')
        
        # Check that each config item was logged
        for key in config.config:
            expected_log = key + '=' + config[key]
            mock_log_instance.add.assert_any_call(expected_log)

    @patch('base.recommender.Log')
    @patch('base.recommender.Data')
    @patch('base.recommender.OptionConf')
    @patch('builtins.print')
    @patch('base.recommender.abspath')
    def test_print_model_info_basic(self, mock_abspath, mock_print, mock_option_conf, mock_data, mock_log):
        """Test print_model_info method basic functionality."""
        config = self.create_mock_config()
        
        mock_abspath.side_effect = lambda x: '/abs' + x
        
        recommender = Recommender(config, [], [], [])
        recommender.print_model_info()
        
        # Check that model information was printed
        mock_print.assert_any_call('Model:', 'TestModel')
        mock_print.assert_any_call('Training Set:', '/abs/path/to/train.txt')
        mock_print.assert_any_call('Test Set:', '/abs/path/to/test.txt')
        mock_print.assert_any_call('Embedding Dimension:', 64)
        mock_print.assert_any_call('Maximum Epoch:', 100)
        mock_print.assert_any_call('Learning Rate:', 0.001)
        mock_print.assert_any_call('Batch Size:', 256)
        mock_print.assert_any_call('Regularization Parameter:', 0.01)

    @patch('base.recommender.Log')
    @patch('base.recommender.Data')
    @patch('base.recommender.OptionConf')
    @patch('builtins.print')
    def test_print_model_info_with_specific_parameters(self, mock_print, mock_option_conf, mock_data, mock_log):
        """Test print_model_info method with specific model parameters."""
        config = self.create_mock_config()
        config.contain.return_value = True  # Model has specific parameters
        config.__getitem__.side_effect = lambda key: {
            'model.name': 'TestModel',
            'item.ranking': 'on -topN 10',
            'embedding.size': '64',
            'num.max.epoch': '100',
            'batch_size': '256',
            'learnRate': '0.001',
            'reg.lambda': '0.01',
            'output.setup': 'on -dir ./output',
            'training.set': '/path/to/train.txt',
            'test.set': '/path/to/test.txt',
            'TestModel': 'on -alpha 0.5 -beta 0.3'
        }[key]
        
        # Mock OptionConf for specific parameters
        mock_args = MagicMock()
        mock_args.keys.return_value = ['-alpha', '-beta']
        mock_args.__getitem__.side_effect = lambda key: {'alpha': '0.5', 'beta': '0.3'}[key[1:]]
        mock_option_conf.side_effect = [MagicMock(), MagicMock(), mock_args]
        
        recommender = Recommender(config, [], [], [])
        recommender.print_model_info()
        
        # Check that specific parameters were printed
        mock_print.assert_any_call('Specific parameters:', 'alpha:0.5  beta:0.3  ')

    @patch('base.recommender.Log')
    @patch('base.recommender.Data')
    @patch('base.recommender.OptionConf')
    def test_abstract_methods_exist(self, mock_option_conf, mock_data, mock_log):
        """Test that abstract methods exist and can be called."""
        config = self.create_mock_config()
        
        recommender = Recommender(config, [], [], [])
        
        # These methods should exist and return None (pass implementation)
        assert recommender.build() is None
        assert recommender.train() is None
        assert recommender.predict('user1') is None
        assert recommender.test() is None
        assert recommender.save() is None
        assert recommender.load() is None
        assert recommender.evaluate('dummy_rec_list') is None

    @patch('base.recommender.Log')
    @patch('base.recommender.Data')
    @patch('base.recommender.OptionConf')
    @patch('builtins.print')
    def test_execute_method_workflow(self, mock_print, mock_option_conf, mock_data, mock_log):
        """Test execute method workflow."""
        config = self.create_mock_config()
        
        recommender = Recommender(config, [], [], [])
        
        # Mock the methods that execute calls
        recommender.initializing_log = MagicMock()
        recommender.print_model_info = MagicMock()
        recommender.build = MagicMock()
        recommender.train = MagicMock()
        recommender.evaluate = MagicMock()
        
        recommender.execute()
        
        # Check that all methods were called in order
        recommender.initializing_log.assert_called_once()
        recommender.print_model_info.assert_called_once()
        recommender.build.assert_called_once()
        recommender.train.assert_called_once()
        recommender.evaluate.assert_called_once()
        
        # Check that progress messages were printed
        mock_print.assert_any_call('Initializing and building model...')
        mock_print.assert_any_call('Training Model...')
        mock_print.assert_any_call('Testing...')
        mock_print.assert_any_call('Evaluating...')

    @patch('base.recommender.Log')
    @patch('base.recommender.Data')
    @patch('base.recommender.OptionConf')
    def test_type_conversions(self, mock_option_conf, mock_data, mock_log):
        """Test that configuration values are converted to correct types."""
        config = self.create_mock_config()
        
        recommender = Recommender(config, [], [], [])
        
        # Check type conversions
        assert isinstance(recommender.emb_size, int)
        assert isinstance(recommender.maxEpoch, int)
        assert isinstance(recommender.batch_size, int)
        assert isinstance(recommender.lRate, float)
        assert isinstance(recommender.reg, float)

    @patch('base.recommender.Log')
    @patch('base.recommender.Data')
    @patch('base.recommender.OptionConf')
    @patch('base.recommender.time')
    @patch('base.recommender.strftime')
    @patch('base.recommender.localtime')
    def test_log_initialization_with_timestamp(self, mock_localtime, mock_strftime, mock_time, mock_option_conf, mock_data, mock_log):
        """Test that log is initialized with correct timestamp."""
        config = self.create_mock_config()
        
        mock_time.return_value = 1234567890
        mock_localtime.return_value = 'mock_time'
        mock_strftime.return_value = '2023-01-01 12-00-00'
        
        recommender = Recommender(config, [], [], [])
        
        # Check that Log was called with correct parameters
        expected_log_name = 'TestModel 2023-01-01 12-00-00'
        mock_log.assert_called_once_with('TestModel', expected_log_name)