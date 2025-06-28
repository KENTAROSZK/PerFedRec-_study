import os
import sys
import pytest
import tempfile
from unittest.mock import patch

# Add the parent directory of PerFedRec++ to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PerFedRec++')))

from data.loader import FileIO

class TestFileIO:
    """Test class for FileIO functionality."""

    def test_write_file_creates_directory(self, tmp_path):
        """Test if write_file creates directories when they don't exist."""
        test_dir = str(tmp_path / "new_dir/")
        test_file = "test.txt"
        content = ["Hello\n", "World\n"]
        
        FileIO.write_file(test_dir, test_file, content)
        
        assert os.path.exists(test_dir)
        assert os.path.exists(test_dir + test_file)
        
        with open(test_dir + test_file, 'r') as f:
            assert f.read() == "Hello\nWorld\n"

    def test_write_file_existing_directory(self, tmp_path):
        """Test write_file when directory already exists."""
        test_dir = str(tmp_path) + "/"
        test_file = "test.txt"
        content = ["Line 1\n", "Line 2\n"]
        
        FileIO.write_file(test_dir, test_file, content)
        
        with open(test_dir + test_file, 'r') as f:
            assert f.read() == "Line 1\nLine 2\n"

    def test_write_file_append_mode(self, tmp_path):
        """Test write_file with append mode."""
        test_dir = str(tmp_path) + "/"
        test_file = "test.txt"
        
        # First write
        FileIO.write_file(test_dir, test_file, ["Line 1\n"])
        # Append write
        FileIO.write_file(test_dir, test_file, ["Line 2\n"], op='a')
        
        with open(test_dir + test_file, 'r') as f:
            content = f.read()
            assert content == "Line 1\nLine 2\n"

    def test_delete_file_existing(self, tmp_path):
        """Test delete_file removes existing file."""
        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        FileIO.delete_file(str(test_file))
        assert not test_file.exists()

    def test_delete_file_nonexistent(self, tmp_path):
        """Test delete_file handles non-existent file gracefully."""
        test_file = str(tmp_path / "nonexistent.txt")
        
        # Should not raise exception
        FileIO.delete_file(test_file)

    def test_load_data_set_graph_type(self, tmp_path):
        """Test load_data_set with graph type data."""
        test_file = tmp_path / "graph_data.txt"
        graph_content = "user1 item1 0.5\nuser2 item2 1.0\nuser3 item3 0.8\n"
        test_file.write_text(graph_content)
        
        data = FileIO.load_data_set(str(test_file), 'graph')
        
        expected = [
            ['user1', 'item1', 0.5],
            ['user2', 'item2', 1.0], 
            ['user3', 'item3', 0.8]
        ]
        assert data == expected

    def test_load_data_set_sequential_type(self, tmp_path):
        """Test load_data_set with sequential type data."""
        test_file = tmp_path / "seq_data.txt"
        seq_content = "seq1:item1 item2 item3\nseq2:item4 item5\n"
        test_file.write_text(seq_content)
        
        data = FileIO.load_data_set(str(test_file), 'sequential')
        
        expected = {
            'seq1': ['item1', 'item2', 'item3'],
            'seq2': ['item4', 'item5']
        }
        assert data == expected

    def test_load_data_set_empty_file(self, tmp_path):
        """Test load_data_set with empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")
        
        data = FileIO.load_data_set(str(test_file), 'graph')
        assert data == []
        
        data = FileIO.load_data_set(str(test_file), 'sequential')
        assert data == {}

    @patch('builtins.print')
    def test_load_user_list(self, mock_print, tmp_path):
        """Test load_user_list functionality."""
        test_file = tmp_path / "users.txt"
        user_content = "user1 extra_data\nuser2\nuser3 more data here\n"
        test_file.write_text(user_content)
        
        user_list = FileIO.load_user_list(str(test_file))
        
        expected = ['user1', 'user2', 'user3']
        assert user_list == expected
        mock_print.assert_called_once_with('loading user List...')

    def test_load_user_list_empty_lines(self, tmp_path):
        """Test load_user_list with empty lines raises IndexError."""
        test_file = tmp_path / "users_with_empty.txt"
        user_content = "user1\n\nuser2\n"
        test_file.write_text(user_content)
        
        # The current implementation doesn't handle empty lines gracefully
        with pytest.raises(IndexError):
            FileIO.load_user_list(str(test_file))

    @patch('builtins.print')
    def test_load_social_data_with_weights(self, mock_print, tmp_path):
        """Test load_social_data with weight values."""
        test_file = tmp_path / "social.txt"
        social_content = "user1 user2 0.5\nuser3 user4 0.8\nuser5 user6 1.0\n"
        test_file.write_text(social_content)
        
        social_data = FileIO.load_social_data(str(test_file))
        
        expected = [
            ['user1', 'user2', 0.5],
            ['user3', 'user4', 0.8],
            ['user5', 'user6', 1.0]
        ]
        assert social_data == expected
        mock_print.assert_called_once_with('loading social data...')

    @patch('builtins.print')
    def test_load_social_data_without_weights(self, mock_print, tmp_path):
        """Test load_social_data without weight values (default to 1)."""
        test_file = tmp_path / "social_no_weights.txt"
        social_content = "user1 user2\nuser3 user4\n"
        test_file.write_text(social_content)
        
        social_data = FileIO.load_social_data(str(test_file))
        
        expected = [
            ['user1', 'user2', 1],
            ['user3', 'user4', 1]
        ]
        assert social_data == expected
        mock_print.assert_called_once_with('loading social data...')

    def test_load_data_set_file_not_found(self):
        """Test load_data_set raises exception for non-existent file."""
        with pytest.raises(FileNotFoundError):
            FileIO.load_data_set("nonexistent_file.txt", 'graph')

    def test_load_user_list_file_not_found(self):
        """Test load_user_list raises exception for non-existent file."""
        with pytest.raises(FileNotFoundError):
            FileIO.load_user_list("nonexistent_file.txt")

    def test_load_social_data_file_not_found(self):
        """Test load_social_data raises exception for non-existent file."""
        with pytest.raises(FileNotFoundError):
            FileIO.load_social_data("nonexistent_file.txt")