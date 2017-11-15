##
# \file run_deconvolution_test.py
#  \brief  Class to test deconvolution script
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date October 2017


import os
import unittest
import SimpleITK as sitk

import pysitk.python_helper as ph

from nsol.definitions import DIR_TMP, DIR_TEST


class RunDeconvolution(unittest.TestCase):

    def setUp(self):
        self.data_2d = os.path.join(DIR_TEST, "2D_Lena_256_blur_noise.png")
        self.result_2d = os.path.join(DIR_TMP, "2D_result.png")

        self.data_3d = os.path.join(DIR_TEST, "3D_SheppLoganPhantom_64.nii.gz")
        self.result_3d = os.path.join(DIR_TMP, "3D_result.nii.gz")

        self.iterations = 5

    def run_reconstruction_2d(self, reconstruction_type):
        cmd_args = []
        cmd_args.append("--observation %s" % self.data_2d)
        cmd_args.append("--result %s" % self.result_2d)
        cmd_args.append("--iterations %d" % self.iterations)
        cmd_args.append("--reconstruction-type %s" % reconstruction_type)
        cmd_args.append("--verbose 0")
        cmd = "nsol_run_deconvolution %s" % (" ").join(cmd_args)
        return ph.execute_command(cmd)

    def run_reconstruction_3d(self, reconstruction_type):
        cmd_args = []
        cmd_args.append("--observation %s" % self.data_3d)
        cmd_args.append("--result %s" % self.result_3d)
        cmd_args.append("--iterations %d" % self.iterations)
        cmd_args.append("--reconstruction-type %s" % reconstruction_type)
        cmd_args.append("--verbose 0")
        cmd = "nsol_run_deconvolution %s" % (" ").join(cmd_args)
        return ph.execute_command(cmd)

    def test_tkl2_deconvolution(self):

        # 2D
        self.assertEqual(self.run_reconstruction_2d("TK0L2"), 0)
        self.assertEqual(self.run_reconstruction_2d("TK1L2"), 0)

        # 3D
        self.assertEqual(self.run_reconstruction_3d("TK0L2"), 0)
        self.assertEqual(self.run_reconstruction_3d("TK1L2"), 0)

    def test_tvl2_deconvolution(self):

        # 2D
        self.assertEqual(self.run_reconstruction_2d("TVL2"), 0)

        # 3D
        self.assertEqual(self.run_reconstruction_3d("TVL2"), 0)

    def test_huberl2_deconvolution(self):

        # 2D
        self.assertEqual(self.run_reconstruction_2d("HuberL2"), 0)

        # 3D
        self.assertEqual(self.run_reconstruction_3d("HuberL2"), 0)
