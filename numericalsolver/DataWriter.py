# \file DataWriter.py
#  \brief Writes numpy arrays to files
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2017


# Import libraries
import os
import sys
import SimpleITK as sitk
import scipy.io
import numpy as np
import re
import natsort

import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh


class DataWriter(object):

    def __init__(self, nda, path_to_file):
        self._nda = nda
        self._path_to_file = path_to_file

        # Get filename extension
        self._file_type = os.path.basename(self._path_to_file).split(".")[1]

        self._write_data = {
            "txt": self._write_data_txt,
            "png": self._write_data_png,
            "mat": self._write_data_mat,
            # "nii": self._write_data_nii,
        }

    def write_data(self):
        self._write_data[self._file_type]()

    def _write_data_png(self):
        ph.write_image(self._nda, self._path_to_file)

    def _write_data_txt(self):
        ph.write_array_to_file(self._path_to_file, self._nda)

    def _write_data_mat(self):
        dic = {"nda": self._nda}
        scipy.io.savemat(self._path_to_file, dic)
        ph.print_info("File '%s' successfully written" %(self._path_to_file))