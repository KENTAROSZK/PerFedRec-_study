
import os
import sys
import pytest

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from data.loader import FileIO

@pytest.fixture
def setup_temp_dir(tmp_path):
    """
    Fixture to create a temporary directory for file operations.
    """
    return tmp_path

def test_write_file_new(setup_temp_dir):
    """
    Test write_file to create a new file.
    """
    test_dir = str(setup_temp_dir) + '/'
    test_file = "new_file.txt"
    content = ["line1\n", "line2\n"]
    FileIO.write_file(test_dir, test_file, content)
    with open(test_dir + test_file, 'r') as f:
        read_content = f.readlines()
    assert read_content == content
    assert os.path.exists(test_dir + test_file)

def test_write_file_append(setup_temp_dir):
    """
    Test write_file to append to an existing file.
    """
    test_dir = str(setup_temp_dir) + '/'
    test_file = "append_file.txt"
    initial_content = ["initial\n"]
    append_content = ["appended\n"]

    # Create initial file
    FileIO.write_file(test_dir, test_file, initial_content)

    # Append to file
    FileIO.write_file(test_dir, test_file, append_content, op='a')

    with open(test_dir + test_file, 'r') as f:
        read_content = f.readlines()
    assert read_content == initial_content + append_content

def test_delete_file_existing(setup_temp_dir):
    """
    Test delete_file for an existing file.
    """
    test_dir = str(setup_temp_dir) + '/'
    test_file = "delete_me.txt"
    # Create a dummy file
    with open(test_dir + test_file, 'w') as f:
        f.write("dummy content")
    assert os.path.exists(test_dir + test_file)

    FileIO.delete_file(test_dir + test_file)
    assert not os.path.exists(test_dir + test_file)

def test_delete_file_non_existent():
    """
    Test delete_file for a non-existent file (should not raise error).
    """
    non_existent_file = "/tmp/non_existent_file_12345.txt"
    FileIO.delete_file(non_existent_file) # Should not raise an error

def test_load_data_set_graph(setup_temp_dir):
    """
    Test load_data_set for 'graph' type.
    """
    test_dir = str(setup_temp_dir) + '/'
    test_file = "graph_data.txt"
    content = [
        "user1 item1 0.5\n",
        "user2 item2 1.0\n",
        "user1 item3 0.8\n"
    ]
    with open(test_dir + test_file, 'w') as f:
        f.writelines(content)

    data = FileIO.load_data_set(test_dir + test_file, 'graph')
    expected_data = [
        ['user1', 'item1', 0.5],
        ['user2', 'item2', 1.0],
        ['user1', 'item3', 0.8]
    ]
    assert data == expected_data

def test_load_data_set_sequential(setup_temp_dir):
    """
    Test load_data_set for 'sequential' type.
    """
    test_dir = str(setup_temp_dir) + '/'
    test_file = "sequential_data.txt"
    content = [
        "seq1:itemA itemB itemC\n",
        "seq2:itemD itemE\n"
    ]
    with open(test_dir + test_file, 'w') as f:
        f.writelines(content)

    data = FileIO.load_data_set(test_dir + test_file, 'sequential')
    expected_data = {
        'seq1': ['itemA', 'itemB', 'itemC'],
        'seq2': ['itemD', 'itemE']
    }
    assert data == expected_data

def test_load_user_list(setup_temp_dir):
    """
    Test load_user_list.
    """
    test_dir = str(setup_temp_dir) + '/'
    test_file = "user_list.txt"
    content = [
        "userA\n",
        "userB some_other_info\n", # Should only take the first part
        "userC\n"
    ]
    with open(test_dir + test_file, 'w') as f:
        f.writelines(content)

    user_list = FileIO.load_user_list(test_dir + test_file)
    expected_list = ["userA", "userB", "userC"]
    assert user_list == expected_list

def test_load_social_data_with_weight(setup_temp_dir):
    """
    Test load_social_data with weights.
    """
    test_dir = str(setup_temp_dir) + '/'
    test_file = "social_data_weighted.txt"
    content = [
        "user1 user2 0.7\n",
        "user3 user4 0.9\n"
    ]
    with open(test_dir + test_file, 'w') as f:
        f.writelines(content)

    social_data = FileIO.load_social_data(test_dir + test_file)
    expected_data = [
        ['user1', 'user2', 0.7],
        ['user3', 'user4', 0.9]
    ]
    assert social_data == expected_data

def test_load_social_data_no_weight(setup_temp_dir):
    """
    Test load_social_data without weights (should default to 1).
    """
    test_dir = str(setup_temp_dir) + '/'
    test_file = "social_data_unweighted.txt"
    content = [
        "userA userB\n",
        "userC userD\n"
    ]
    with open(test_dir + test_file, 'w') as f:
        f.writelines(content)

    social_data = FileIO.load_social_data(test_dir + test_file)
    expected_data = [
        ['userA', 'userB', 1],
        ['userC', 'userD', 1]
    ]
    assert social_data == expected_data
