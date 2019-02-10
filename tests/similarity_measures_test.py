##
# \file similarity_measures_test.py
#  \brief  Class containing unit tests for lossFunctions
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Sept 2017

import os
import numpy as np
import unittest
import sys
import matplotlib.pyplot as plt
import pysitk.python_helper as ph

from nsol.similarity_measures import SimilarityMeasures as sim_meas
from nsol.definitions import DIR_TEST


class SimilarityMeasuresTest(unittest.TestCase):

    def setUp(self):
        self.accuracy = 4
        filename = os.path.join(DIR_TEST, "2D_BrainWeb.png")
        self.image = np.array(ph.read_image(filename), dtype=np.float64)
        self.image_scale = self.image * 2
        self.image_off = self.image + 2

        self.x = self.image.flatten()
        self.x_scale = self.image_scale.flatten()
        self.x_off = self.image_off.flatten()

    def test_absolute_errors(self):
        diff = sim_meas.mean_absolute_error(self.x, self.x_off)
        self.assertAlmostEqual(
            diff - np.abs(self.x - self.x_off).mean(), 0, places=self.accuracy)

    def test_squared_errors(self):

        diff = sim_meas.sum_of_squared_differences(self.x, self.x_off)
        self.assertAlmostEqual(
            diff - np.sum(np.square(self.x - self.x_off)), 0,
            places=self.accuracy)

        diff = sim_meas.mean_squared_error(self.x, self.x_off)
        self.assertAlmostEqual(
            diff - np.square(self.x - self.x_off).mean(), 0,
            places=self.accuracy)

        diff = sim_meas.sum_of_squared_differences(self.x, self.x)
        self.assertEqual(np.around(
            diff, decimals=self.accuracy), 0)

        diff = sim_meas.sum_of_squared_differences(self.x, self.x_off)
        error = self.x.size * 4
        self.assertEqual(np.around(
            abs(diff - error), decimals=self.accuracy), 0)

    def test_peak_signal_to_noise_ratio(self):
        diff = sim_meas.peak_signal_to_noise_ratio(self.x, self.x)
        self.assertEqual(np.around(
            diff, decimals=self.accuracy), np.inf)

    def test_normalized_cross_correlation(self):
        diff = sim_meas.normalized_cross_correlation(self.x, self.x)
        self.assertEqual(np.around(
            abs(diff - 1), decimals=self.accuracy), 0)

        diff = sim_meas.normalized_cross_correlation(self.x, -self.x)
        self.assertEqual(np.around(
            abs(diff + 1), decimals=self.accuracy), 0)

        diff = sim_meas.normalized_cross_correlation(self.x, self.x_off)
        self.assertEqual(np.around(
            abs(diff - 1), decimals=self.accuracy), 0)

        diff = sim_meas.normalized_cross_correlation(self.x, self.x_scale)
        self.assertEqual(np.around(
            abs(diff - 1), decimals=self.accuracy), 0)

    def test_dice_score(self):

        # Create naive mask
        x_mask = np.zeros_like(self.x, dtype=bool)
        x_mask[np.where(self.x > 100)] = True

        # ph.show_array(x_mask.reshape(self.image.shape))

        dice = sim_meas.dice_score(x_mask, x_mask)
        self.assertEqual(np.around(
            abs(dice - 1), decimals=self.accuracy), 0)

        dice = sim_meas.dice_score(x_mask, np.zeros_like(x_mask))
        self.assertEqual(np.around(
            dice, decimals=self.accuracy), 0)
