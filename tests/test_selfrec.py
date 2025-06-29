

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from SELFRec import SELFRec
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

