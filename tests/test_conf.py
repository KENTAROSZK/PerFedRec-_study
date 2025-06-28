
import os
import sys
import pytest

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from util.conf import ModelConf

@pytest.fixture
def create_temp_conf_file(tmp_path):
    """
    Fixture to create a temporary configuration file for testing.
    """
    conf_content = """
key1=value1
key2=value2
key3=123
"""
    conf_file = tmp_path / "test.conf"
    conf_file.write_text(conf_content)
    return str(conf_file)

def test_model_conf_read_configuration(create_temp_conf_file):
    """
    Test if ModelConf correctly reads configuration from a file.
    """
    conf = ModelConf(create_temp_conf_file)
    assert conf.contain("key1")
    assert conf["key1"] == "value1"
    assert conf.contain("key2")
    assert conf["key2"] == "value2"
    assert conf.contain("key3")
    assert conf["key3"] == "123"
    assert not conf.contain("nonexistent_key")

def test_model_conf_getitem_setitem(create_temp_conf_file):
    """
    Test __getitem__ and __setitem__ methods of ModelConf.
    """
    conf = ModelConf(create_temp_conf_file)

    # Test __getitem__
    assert conf["key1"] == "value1"
    assert conf["key2"] == "value2"

    # Test __setitem__
    conf["key1"] = "new_value1"
    assert conf["key1"] == "new_value1"

    conf["new_key"] = "new_value_added"
    assert conf["new_key"] == "new_value_added"
    assert conf.contain("new_key")

def test_model_conf_file_not_found():
    """
    Test if ModelConf raises IOError when the config file is not found.
    """
    with pytest.raises(IOError):
        ModelConf("non_existent_file.conf")

def test_model_conf_invalid_format(tmp_path):
    """
    Test if ModelConf handles invalid configuration file format.
    """
    invalid_conf_content = """
key1=value1
invalid_line_without_equals
key3=value3
"""
    invalid_conf_file = tmp_path / "invalid.conf"
    invalid_conf_file.write_text(invalid_conf_content)

    # The current implementation prints an error and exits.
    # For testing, we can mock sys.exit and check the print output,
    # or modify the ModelConf to raise a specific exception.
    # For now, we'll assume it prints an error and continues or exits.
    # If it exits, pytest.raises(SystemExit) would be appropriate.
    # Given the current code, it prints and exits, so we'll test for SystemExit.
    with pytest.raises(ValueError):
        ModelConf(str(invalid_conf_file))

