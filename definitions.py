import os
import sys

DIR_VOLUMETRIC_RECONSTRUCTION = os.path.abspath(os.environ['VOLUMETRIC_RECONSTRUCTION_DIR'])
DIR_TEST = os.path.join(DIR_VOLUMETRIC_RECONSTRUCTION, 'test-data/')
DIR_TMP = "/tmp/"

# Linked executables
ITKSNAP_EXE = "itksnap"
FSLVIEW_EXE = "fslview"
NIFTYVIEW_EXE = "NiftyView"
BET_EXE = "bet"
REG_ALADIN_EXE = "reg_aladin"
REG_F3D_EXE = "reg_f3d"

# Constants
EPS = 1e-10

ALLOWED_EXTENSIONS = ["nii.gz", "nii"]
REGEX_FILENAMES = "[A-Za-z0-9+-_]+"
REGEX_FILENAME_EXTENSIONS = "(" + "|".join(ALLOWED_EXTENSIONS) + ")"

info = {
    "name": "Volumetric MRI Reconstruction from 2D Slices "
    "in the Presence of Motion",
    "version": "0.1.0",
    "description": "",
    "web_info": "",
    "repository": {
        "type": "",
        "url": "https://cmiclab.cs.ucl.ac.uk/mebner/VolumetricReconstruction"
    },
    "authors": "Michael Ebner",
    "dependencies": {
        # requirements.txt file automatically generated using pipreqs.
        # "python" : "{0}/requirements.txt".format(DIR_ROOT)
        # pip install -r requirements.txt before running the
        # code.
    }
}
