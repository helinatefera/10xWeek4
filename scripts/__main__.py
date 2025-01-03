"""
High-level script for running the project.
"""

import argparse
import sys

from loguru import logger

from .setup_data import extract_data

# Setup the logger
logger.remove(0)
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.configure(
    handlers=[
        {
            "sink": sys.stdout,
            "format": LOG_FORMAT,
            "colorize": True,
            "backtrace": True,
            "diagnose": True,
        }
    ]
)

parser = argparse.ArgumentParser(
    description="High-level script for running the project."
)
parser.add_argument(
    "--extract-data",
    action="store_true",
    help="Extract the data from the given zip file.",
)
parser.add_argument("--data-path", type=str, help="The path to the zip file.")
parser.add_argument(
    "--output-path", type=str, help="The path to extract the data to."
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Increase the verbosity of the logger.",
)

args = parser.parse_args()

if args.verbose:
    logger.level("DEBUG")

# Dependency logic
if args.extract_data:
    if not args.data_path:
        parser.error(
            "--data-path is required when --extract-data is specified."
        )
    if not args.output_path:
        parser.error(
            "--output-path is required when --extract-data is specified."
        )
    extract_data(args.data_path, args.output_path)
