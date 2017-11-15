##
# \file run_denoising_study_test.py
#  \brief  Class to test denoising study script
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date October 2017


import os
import unittest
import SimpleITK as sitk

import pysitk.python_helper as ph

from nsol.definitions import DIR_TMP, DIR_TEST


class RunDenoisingStudy(unittest.TestCase):

    def setUp(self):
        self.data_2d = os.path.join(DIR_TEST, "2D_Lena_256_blur_noise.png")
        self.ref_2d = os.path.join(DIR_TEST, "2D_Lena_256_blur_noise.png")
        self.output_2d = os.path.join(DIR_TMP, "param_study_2d")

        self.data_3d = os.path.join(DIR_TEST, "3D_SheppLoganPhantom_64.nii.gz")
        self.ref_3d = os.path.join(DIR_TEST, "3D_SheppLoganPhantom_64.nii.gz")
        self.output_3d = os.path.join(DIR_TMP, "param_study_3d")

        self.iterations = 5
        self.alpha_range = [0.01, 0.05, 2]

    def run_reconstruction_2d(self, reconstruction_type):
        cmd_args = []
        cmd_args.append("--observation %s" % self.data_2d)
        cmd_args.append("--dir-output %s" % self.output_2d)
        cmd_args.append("--iterations %d" % self.iterations)
        cmd_args.append("--reconstruction-type %s" % reconstruction_type)
        cmd_args.append("--alpha-range %s" %
                        (" ").join([str(a) for a in self.alpha_range]))
        cmd_args.append("--verbose 0")
        cmd = "nsol_run_denoising_study %s" % (" ").join(cmd_args)
        return ph.execute_command(cmd)

    def run_reconstruction_3d(self, reconstruction_type):
        cmd_args = []
        cmd_args.append("--observation %s" % self.data_3d)
        cmd_args.append("--dir-output %s" % self.output_3d)
        cmd_args.append("--iterations %d" % self.iterations)
        cmd_args.append("--reconstruction-type %s" % reconstruction_type)
        cmd_args.append("--alpha-range %s" %
                        (" ").join([str(a) for a in self.alpha_range]))
        cmd_args.append("--verbose 0")
        cmd = "nsol_run_denoising_study %s" % (" ").join(cmd_args)
        return ph.execute_command(cmd)

    def test_tv_denoising(self):

        # 2D
        self.assertEqual(self.run_reconstruction_2d("TVL1"), 0)
        self.assertEqual(self.run_reconstruction_2d("TVL2"), 0)

        # 3D
        self.assertEqual(self.run_reconstruction_3d("TVL1"), 0)
        self.assertEqual(self.run_reconstruction_3d("TVL2"), 0)
