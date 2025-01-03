"""
This script is used to  extract the data from the given URL.
"""

import pathlib
import zipfile


def extract_data(filepath="./data/data.zip", output_path=".") -> None:
    """
    Extracts the data from the given zip file to the given path.

    :param filepath: The path to the zip file.
    :param output_path: The path to extract the data to.

    :return: None
    """
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(output_path)


if __name__ == "__main__":
    extract_data()
