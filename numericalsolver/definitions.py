import os
import sys

DIR_ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_TEST = os.path.join(DIR_ROOT, "..", "data")

# Constants
EPS = 1e-10

REGEX_FILENAMES = "[A-Za-z0-9+-_]+"
FILENAME_EXTENSION = "txt"