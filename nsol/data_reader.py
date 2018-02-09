# \file DataReader.py
#  \brief Reads data and returns numpy arrays
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

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh


class DataReader(object):

    def __init__(self, path_to_file):
        self._path_to_file = path_to_file

        # Get filename extension
        self._file_type = os.path.basename(self._path_to_file).split(".")[1]

        self._read_data = {
            "png": self._read_data_png,
            "mat": self._read_data_mat,
            "nii": self._read_data_nii,
        }

        self._nda = None
        self._image_sitk = None

    def read_data(self):
        if not ph.file_exists(self._path_to_file):
            raise IOError("Filename '%s' not found" % (self._path_to_file))

        self._read_data[self._file_type]()

    def get_data(self):
        return np.array(self._nda, dtype=np.float64)

    def get_image_sitk(self):
        return self._image_sitk

    def _read_data_png(self):
        self._nda = ph.read_image(self._path_to_file)

    def _read_data_mat(self):
        dic = scipy.io.loadmat(self._path_to_file)
        ndas = [dic[k] for k in dic.keys() if isinstance(dic[k], np.ndarray)]

        if len(ndas) > 1:
            raise IOError("MAT file '%s' must include one array only" %
                          (self._path_to_file))

        self._nda = ndas[0]

    def _read_data_nii(self):
        self._image_sitk = sitkh.read_nifti_image_sitk(
            self._path_to_file)
        self._nda = sitk.GetArrayFromImage(self._image_sitk)
