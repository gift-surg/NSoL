##
# \file noise.py
# \brief      Class to add different kinds of noise
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

# Import libraries
import numpy as np


##
# Class to apply noise
# \date       2017-07-22 23:41:01+0100
#
class Noise(object):

    ##
    # Store data array where noise shall be applied
    # \date       2017-07-22 23:40:02+0100
    #
    # \param      self  The object
    # \param      data  Data as numpy array
    # \param      seed  The seed
    #
    def __init__(self, data, seed=None):
        np.random.seed(seed=seed)
        self._data = np.array(data)

    ##
    # Gets the noisy data.
    # \date       2017-07-22 23:41:17+0100
    #
    # \param      self  The object
    #
    # \return     The noisy data as numpy array
    #
    def get_noisy_data(self):
        return self._data

    ##
    # Adds additive Gaussian noise.
    # \date       2017-07-22 23:41:29+0100
    #
    # \param      self         The object
    # \param      noise_level  The noise level between 0 and 1
    # \param      mean         Mean/centre for Gaussian distribution
    # \param      sigma        Standard deviation for Gaussian
    #
    def add_gaussian_noise(self, noise_level=0.01, mean=0, sigma=1):
        self._data += noise_level * self._data.max() * \
            np.random.normal(size=self._data.shape, loc=mean, scale=sigma)

    ##
    # Adds additive Poisson noise.
    # \date       2017-07-22 23:41:29+0100
    #
    # \param      self         The object
    # \param      noise_level  The noise level between 0 and 1
    # \param      lmbda        Expectation of interval, float >= 0
    #
    def add_poisson_noise(self, noise_level=0.01, lmbda=1):
        self._data += noise_level * self._data.max() * \
            np.random.poisson(size=self._data.shape, lam=lmbda)

    ##
    # Adds an uniform noise.
    # \date       2017-07-22 23:42:11+0100
    #
    # \param      self         The object
    # \param      noise_level  The noise level between 0 and 1
    #
    def add_uniform_noise(self, noise_level=0.01):
        self._data += noise_level * self._data.max() * \
            np.random.rand(*self._data.shape)

    ##
    # Replace random pixels with zero (pepper) and one (salt)
    # \date       2017-07-22 22:51:41+0100
    #
    # \param      data          numpy array
    #
    # \return     noised numpy array
    #
    def add_salt_and_pepper_noise(self, salt_vs_pepper=0.5, amount=0.1):

        # Define value for salt and pepper
        val_salt = self._data.max()
        val_pepper = self._data.min()

        shape = self._data.shape
        self._data = self._data.flatten()

        # Pick size amount of random indices from data array
        size = int(amount * self._data.size)
        samples_all = np.random.choice(
            np.arange(0, self._data.size),
            size=size,
            replace=False)

        # Chose the first part to be of salt value
        N_white = int(salt_vs_pepper * samples_all.size)

        self._data[samples_all[0:N_white]] = val_salt
        self._data[samples_all[N_white:]] = val_pepper

        # Reshape to original shape
        self._data = self._data.reshape(*shape)
