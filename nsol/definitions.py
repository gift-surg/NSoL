import os
import sys

from pysitk.definitions import DIR_TMP

DIR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_TEST = os.path.join(DIR_ROOT, "data")

# Constants
EPS = 1e-10

REGEX_FILENAMES = "[A-Za-z0-9+-_]+"
FILENAME_EXTENSION = "txt"

ALLOWED_INPUT_FILE_EXTENSIONS = ["mat", "png", "nii", "nii.gz"]
ALLOWED_NOISE_TYPES = ["gaussian", "poisson", "s&p", "uniform"]
