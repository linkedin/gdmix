import json
import os
import tempfile
import unittest
import shutil
from gdmixworkflow.common.utils import *


class TestUtils(unittest.TestCase):
    """
    Test gdmix workflow utils
    """

    def setUp(self):
        """
        Set the configuration. json.

        Args:
            self: (todo): write your description
        """
        self.output_dir = tempfile.mkdtemp()
        config = {"a": {"b1": "b2"},
                  "c": "d"}
        self.config_file_name = os.path.join(self.output_dir, "config.json")
        with open(self.config_file_name, 'w') as f:
            json.dump(config, f)

    def tearDown(self):
        """
        Tear down the output directory.

        Args:
            self: (todo): write your description
        """
        shutil.rmtree(self.output_dir)

    def test_gen_random_string(self):
        """
        Generates a random string.

        Args:
            self: (todo): write your description
        """
        expectedLen = 8
        actualLen = len(gen_random_string(expectedLen))

        self.assertEqual(actualLen, expectedLen)

    def test_abbr(self):
        """
        Test for the test case.

        Args:
            self: (todo): write your description
        """
        inputStr = "fixed-effect"
        expected = "f10t"
        actual = abbr(inputStr)
        self.assertEqual(actual, expected)

    def test_json_config_file_to_obj(self):
        """
        Test if a json config file.

        Args:
            self: (todo): write your description
        """
        config_obj = json_config_file_to_obj(self.config_file_name)
        self.assertEqual(config_obj.a.b1, "b2")
        self.assertEqual(config_obj.c, "d")

    def test_flatten_config_obj(self):
        """
        Flatten a config file.

        Args:
            self: (todo): write your description
        """
        config_obj = json_config_file_to_obj(self.config_file_name)
        actual = {}
        flatten_config_obj(actual, config_obj)
        expected = {"b1": "b2", "c": "d"}
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
