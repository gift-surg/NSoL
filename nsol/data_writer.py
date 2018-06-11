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

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh


class DataWriter(object):

    ##
    # Store relevant reader information
    # \date       2017-09-13 12:00:29+0100
    #
    # \param      self          The object
    # \param      nda           numpy data array
    # \param      path_to_file  Path to write image
    # \param      image_sitk    sitk.Image object for (optional) nii header in
    #                           case of writing to nii/nii.gz file
    #
    def __init__(self, nda, path_to_file, image_sitk=None):
        self._nda = nda
        self._path_to_file = path_to_file
        self._image_sitk = image_sitk

        # Get filename extension
        self._file_type = os.path.basename(self._path_to_file).split(".")[1]

        self._write_data = {
            "txt": self._write_data_txt,
            "png": self._write_data_png,
            "mat": self._write_data_mat,
            "nii": self._write_data_nii,
        }

    def write_data(self):
        ph.create_directory(os.path.dirname(self._path_to_file))
        self._write_data[self._file_type]()
        ph.print_info("File written to '%s'" % self._path_to_file)

    def _write_data_png(self):
        nda = np.round(np.array(self._nda)).astype(np.uint8)
        ph.write_image(nda, self._path_to_file)

    def _write_data_txt(self):
        ph.write_array_to_file(self._path_to_file, self._nda)

    def _write_data_mat(self):
        dic = {"nda": self._nda}
        scipy.io.savemat(self._path_to_file, dic)
        ph.print_info("File written to '%s'" % (self._path_to_file))

    def _write_data_nii(self):
        image_sitk = sitk.GetImageFromArray(self._nda)

        if self._image_sitk is not None:
            image_sitk.CopyInformation(self._image_sitk)

        sitkh.write_nifti_image_sitk(image_sitk, self._path_to_file)
