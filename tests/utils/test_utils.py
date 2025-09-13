import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from grasp.utils import constants, utils

sys.path.append(str(Path(__file__).parent.parent.parent))


class TestToolUtils(unittest.TestCase):
    def test_extract_and_load_json(self):
        test_text = 'Here is some text {"key": "value"} and more text'
        result = utils.extract_and_load_json(test_text)
        assert result == {"key": "value"}

    def test_load_json(self):
        test_text = '{"key": "value"}'
        result = utils.load_json(test_text)
        assert result == {"key": "value"}

    def test_validate_required_keys(self):
        config = {"key1": "value1", "key2": "value2"}
        utils.validate_required_keys(["key1", "key2"], config, "test")

        with pytest.raises(ValueError):
            utils.validate_required_keys(["key1", "key3"], config, "test")

    def test_get_func_from_str(self):
        func = utils.get_func_from_str("os.path.exists")
        assert callable(func)
        assert func == os.path.exists

    def test_flatten_dict(self):
        nested_dict = {"a": 1, "b": {"c": 2, "d": 3}, "e": {"f": {"g": 4}}}
        result = utils.flatten_dict(nested_dict)
        assert result == {"a": 1, "b_c": 2, "b_d": 3, "e_f_g": 4}

    def test_deep_update(self):
        target = {"a": 1, "b": {"c": 2}}
        source = {"b": {"d": 3}, "e": 4}
        utils.deep_update(target, source)
        assert target == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

    def test_convert_messages_from_chat_format_to_langchain(self):
        msg = [
            {"role": "system", "content": "test system msg"},
            {"role": "user", "content": "test user msg"},
            {"role": "assistant", "content": "test assistant msg"},
        ]
        omsg = utils.convert_messages_from_chat_format_to_langchain(msg)
        assert len(omsg) == 3

    def test_convert_messages_from_config_to_chat_format(self):
        msg = [
            {"system": "test system msg"},
            {"user": "test user msg"},
            {"assistant": "test assistant msg"},
        ]
        omsg = utils.convert_messages_from_config_to_chat_format(msg)
        assert len(omsg) == 3 and omsg[0]["role"] == "system"

    def test_extract_pattern(self):
        test_string = "Hello 123 world 456"
        pattern = r"\d+"
        result = utils.extract_pattern(test_string, pattern)
        assert result == ["123", "456"]

    @patch("grasp.core.dataset.huggingface_handler.HuggingFaceHandler.read")
    def test_get_dataset(self, hf_read):
        dataset_mocked = [
            {"code": "my python code"},
            {"code": "my java code"},
            {"code": "my c++ code"},
        ]
        # set return for 4 calls
        hf_read.side_effect = [
            dataset_mocked,
            dataset_mocked,
            dataset_mocked,
            dataset_mocked,
        ]
        data_src = {
            "type": "hf",
            "repo_id": "google-research-datasets/mbpp",
            "config_name": "sanitized",
            "split": ["train"],
        }
        obj = utils.get_dataset(data_src)
        assert isinstance(obj, list) and len(obj) > 0

    @patch("grasp.utils.utils.get_dataset")
    def test_fetch_next_record(self, mock_get_dataset):
        data_src = {
            "type": "hf",
            "repo_id": "google-research-datasets/mbpp",
            "config_name": "sanitized",
            "split": ["train"],
        }
        complete_dataset = [
            {"code": "my python code"},
            {"code": "my java code"},
            {"code": "my c++ code"},
        ]
        # set return for each call
        mock_get_dataset.side_effect = [
            complete_dataset,
            complete_dataset,
            complete_dataset,
        ]
        # fetch 2 records
        rec1 = utils.fetch_next_record(data_src, "code")
        rec2 = utils.fetch_next_record(data_src, "code")
        assert "python" in rec1 and "java" in rec2

    @patch.dict(os.environ, {}, clear=True)
    @patch("grasp.utils.utils.load_yaml_file")
    def test_load_model_config_url_list(self, mock_load_yaml):
        """Test that pipe-separated URLs in environment variables are correctly parsed into lists."""
        # Mock the base configs loaded from YAML
        mock_load_yaml.return_value = {
            "model1": {"model_type": "vllm", "parameters": {"temperature": 0.7}},
            "model2": {
                "model_type": "azure_openai",
                "parameters": {"temperature": 1.0},
            },
        }

        # Set up environment variables with pipe-separated URLs
        os.environ["GRASP_MODEL1_URL"] = (
            f"http://server1.example.com/v1/{constants.LIST_SEPARATOR}http://server2.example.com/v1/"
        )
        os.environ["GRASP_MODEL1_TOKEN"] = "test-token-1"

        # Set up a regular URL without separator
        os.environ["GRASP_MODEL2_URL"] = "http://api.openai.com/v1/"
        os.environ["GRASP_MODEL2_TOKEN"] = "test-token-2"

        # Call the function
        result = utils.load_model_config()

        # Verify model1 has a list of URLs
        self.assertIsInstance(result["model1"]["url"], list)
        self.assertEqual(len(result["model1"]["url"]), 2)
        self.assertEqual(result["model1"]["url"][0], "http://server1.example.com/v1/")
        self.assertEqual(result["model1"]["url"][1], "http://server2.example.com/v1/")
        self.assertEqual(result["model1"]["auth_token"], "test-token-1")

        # Verify model2 has a single URL string
        self.assertIsInstance(result["model2"]["url"], str)
        self.assertEqual(result["model2"]["url"], "http://api.openai.com/v1/")
        self.assertEqual(result["model2"]["api_key"], "test-token-2")


if __name__ == "__main__":
    unittest.main()
