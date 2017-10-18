##
# \file prior_measures.py
# \brief      Collection of prior measures, i.e. regularizer costs
#
# \todo proper unit testing of these functions to be done yet
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#


import numpy as np

from nsol.loss_functions import LossFunctions as loss_fun


class PriorMeasures(object):

    @staticmethod
    def zeroth_order_tikhonov(x):
        return 0.5 * np.sum(np.square(x))

    @staticmethod
    def first_order_tikhonov(x, D):
        return 0.5 * np.sum(np.square(D(x)))

    @staticmethod
    def total_variation(x, D, dimension):
        Dx_split = np.array_split(D(x), dimension)

        # Dx_i ** 2
        sum_Dx_i_squared = Dx_split[0]**2

        # Compute sum_k x_k^2
        for i in range(1, len(Dx_split)):
            sum_Dx_i_squared += Dx_split[i]**2

        return np.sum(np.sqrt(sum_Dx_i_squared))

    @staticmethod
    def huber(x, D, dimension, gamma=0.05):
        Dx_split = np.array_split(D(x), dimension)

        # Dx_i ** 2
        sum_Dx_i_squared = Dx_split[0]**2

        # Compute sum_k x_k^2
        for i in range(1, len(Dx_split)):
            sum_Dx_i_squared += Dx_split[i]**2

        Dx_gamma = loss_fun.huber(sum_Dx_i_squared, gamma=gamma) / (2.*gamma)

        return np.sum(Dx_gamma)
