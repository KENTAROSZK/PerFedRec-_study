
import os
import sys
import pytest

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from util.conf import ModelConf, OptionConf

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

def test_option_conf_main_option_on():
    """
    Test OptionConf with 'on' as the main option.
    """
    option_str = "on -key1 val1 -key2 val2"
    conf = OptionConf(option_str)
    assert conf.is_main_on()
    assert conf.contain("-key1")
    assert conf["-key1"] == "val1"
    assert conf.contain("-key2")
    assert conf["-key2"] == "val2"

def test_option_conf_main_option_off():
    """
    Test OptionConf with 'off' as the main option.
    """
    option_str = "off -keyA valueA -keyB valueB"
    conf = OptionConf(option_str)
    assert not conf.is_main_on()
    assert conf.contain("-keyA")
    assert conf["-keyA"] == "valueA"
    assert conf.contain("-keyB")
    assert conf["-keyB"] == "valueB"

def test_option_conf_multiple_values():
    """
    Test OptionConf with options having multiple values.
    """
    option_str = "on -path /usr/local/bin -name myapp"
    conf = OptionConf(option_str)
    assert conf.is_main_on()
    assert conf.contain("-path")
    assert conf["-path"] == "/usr/local/bin"
    assert conf.contain("-name")
    assert conf["-name"] == "myapp"

def test_option_conf_no_value():
    """
    Test OptionConf with an option having no explicit value (should default to 1).
    """
    option_str = "on -flag"
    conf = OptionConf(option_str)
    assert conf.is_main_on()
    assert conf.contain("-flag")
    assert conf["-flag"] == "1"

def test_option_conf_keys_method():
    """
    Test the keys() method of OptionConf.
    """
    option_str = "on -key1 val1 -key2 val2"
    conf = OptionConf(option_str)
    expected_keys = {"-key1", "-key2"}
    assert set(conf.keys()) == expected_keys

def test_option_conf_getitem_invalid_key():
    """
    Test __getitem__ with an invalid key, expecting SystemExit.
    """
    option_str = "on -key1 val1"
    conf = OptionConf(option_str)
    with pytest.raises(SystemExit):
        _ = conf["-nonexistent"]

