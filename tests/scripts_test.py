"""
Test module for the scripts
"""

import os
import sys
import unittest
import zipfile

sys.path.append("..")

from scripts.setup_data import extract_data  # noqa: E402


class TestExtractData(unittest.TestCase):
    """
    Test case for the extract_data function
    """

    def setUp(self):
        """
        Set up the test environment
        """
        self.test_zip = "test_data.zip"
        self.test_output_dir = "test_output"
        with zipfile.ZipFile(self.test_zip, "w") as zipf:
            zipf.writestr("dummy_file.txt", "This is a test file.")

    def tearDown(self):
        """
        Clean up the test environment
        """
        if os.path.exists(self.test_zip):
            os.remove(self.test_zip)
        if os.path.exists(self.test_output_dir):
            for root, dirs, files in os.walk(
                self.test_output_dir, topdown=False
            ):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.test_output_dir)

    def test_extract_data(self):
        """
        Test the extract_data function
        """
        extract_data(filepath=self.test_zip, output_path=self.test_output_dir)

        self.assertTrue(os.path.exists(self.test_output_dir))

        extracted_file_path = os.path.join(
            self.test_output_dir, "dummy_file.txt"
        )
        self.assertTrue(os.path.exists(extracted_file_path))

        with open(extracted_file_path, "r", encoding="utf-8") as file:
            content = file.read()
        self.assertEqual(content, "This is a test file.")


if __name__ == "__main__":
    unittest.main()
