##
# \file kernels.py
# \brief      Class to create kernels used for blurring and differentiation in
#             1D, 2D and 3D
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

# Import libraries
import numpy as np
from abc import ABCMeta, abstractmethod


class Kernels(object):
    __metaclass__ = ABCMeta

    def __init__(self, dimension, spacing):

        spacing = np.atleast_1d(spacing).astype(float)

        if spacing.size != dimension:
            raise ValueError("dimension of spacing and space must be the same")

        self._dimension = dimension
        self._spacing = spacing

    def get_dimension(self):
        return self._dimension

    def get_spacing(self):
        return self._spacing

    ##
    # Gets the Gaussian kernel created from a given covariance matrix and
    # cut-off distance.
    # \date       2017-07-19 12:48:06+0100
    #
    # \param      self       The object
    # \param      cov        Variance covariance matrix as numpy array
    # \param      alpha_cut  Cut-off distance in integer, i.e. 3 means cutting
    #                        off a 3 sigma in each direction
    # \param      spacing    Image spacing in x-, y- (z-) direction as numpy
    #                        array
    #
    # \return     The Gaussian kernel with its size depending on alpha_cut
    #
    @abstractmethod
    def get_gaussian(self):
        pass

    ##
    # Get kernel to compute derivative in x-direction based on forward
    # difference.
    #
    # Kernel is assumed to be applied on arrays of the form array =
    # array[(z,)y,x], i.e. the 'correct' direction by viewing the resulting
    # nifti-image differentiation. Resulting kernel can be used via
    # ndimage.convolve(nda, kernel)
    # \date       2017-07-19 15:13:04+0100
    #
    # \param      self     The object
    #
    # \return     kernel to differentiate array in x-direction.
    #
    @abstractmethod
    def get_dx_forward_difference(self):
        pass

    @abstractmethod
    def get_dx_backward_difference(self):
        pass


class Kernels1D(Kernels):

    def __init__(self, spacing=1):
        super(self.__class__, self).__init__(dimension=1, spacing=spacing)

    def get_gaussian(self, cov, alpha_cut=3):

        # Generate intervals for x based on cut-off distance given by
        # standard deviation, alpha_cut and spacing
        x_max = np.ceil(np.sqrt(cov) * alpha_cut/self._spacing)

        step = 1
        points = np.arange(-x_max, x_max+step, step)

        # Mean/Origin for Gaussian blurring
        origin = 0.

        # Compute scaled, inverse covariance matrix
        cov_scale_inv = self._spacing**2 / cov

        # Compute Gaussian weights
        values = (points-origin)*cov_scale_inv * (points-origin)
        kernel = np.exp(-0.5 * values)
        kernel = kernel/np.sum(kernel)

        return kernel

    def get_dx_forward_difference(self):
        kernel = np.zeros(2)
        kernel = np.array([1, -1])

        return kernel / self._spacing

    def get_dx_backward_difference(self):
        kernel = np.zeros(3)
        kernel = np.array([0, 1, -1])

        return kernel / self._spacing


class Kernels2D(Kernels):

    def __init__(self, spacing=np.ones(2)):
        super(self.__class__, self).__init__(dimension=2, spacing=spacing)

    def get_gaussian(self, cov, alpha_cut=3):

        if cov.shape != (self._dimension, self._dimension):
            raise ValueError("Numpy array 'cov' must be of shape (%d,%d)" %
                             (self._dimension, self._dimension))

        # Generate intervals for x and y based on cut-off distance given by
        # standard deviation, alpha_cut and spacing
        [x_max, y_max] = np.ceil(
            np.sqrt(cov.diagonal()) * alpha_cut/self._spacing)

        step = 1
        x_interval = np.arange(-x_max, x_max+step, step)
        y_interval = np.arange(-y_max, y_max+step, step)

        # Generate arrays of 2D points bearing in mind that nifti-nda.shape =
        # (y,x)-coord. 'ij' yields vertical x-coordinate for image
        [X, Y] = np.meshgrid(x_interval, y_interval,
                             indexing='ij')
        points = np.array([Y.flatten(), X.flatten()])

        # Mean/Origin for Gaussian blurring
        origin = np.zeros((self._dimension, 1))

        # Scaling matrix depending on spacing
        S = np.diag(self._spacing)

        # Compute scaled, inverse covariance matrix
        cov_scale_inv = S.dot(np.linalg.inv(cov)).dot(S)

        # Compute Gaussian weights
        values = np.sum((points-origin)*cov_scale_inv.dot(points-origin), 0)
        kernel = np.exp(-0.5 * values)
        kernel = kernel/np.sum(kernel)

        # Reshape kernel
        kernel = kernel.reshape(x_interval.size, y_interval.size)

        return kernel

    def get_dx_forward_difference(self):

        # kernel = np.zeros((y,x))
        kernel = np.zeros((1, 2))
        kernel[:] = np.array([1, -1])

        return kernel / self._spacing[0]

    def get_dx_backward_difference(self):

        # kernel = np.zeros((y,x))
        kernel = np.zeros((1, 3))
        kernel[:] = np.array([0, 1, -1])

        return kernel / self._spacing[0]

    def get_dy_forward_difference(self):

        # kernel = np.zeros((y,x))
        kernel = np.zeros((2, 1))
        kernel[:] = np.array([[1], [-1]])

        return kernel / self._spacing[1]

    def get_dy_backward_difference(self):

        # kernel = np.zeros((y,x))
        kernel = np.zeros((3, 1))
        kernel[:] = np.array([[0], [1], [-1]])

        return kernel / self._spacing[1]


class Kernels3D(Kernels):

    def __init__(self, spacing=np.ones(3)):
        super(self.__class__, self).__init__(dimension=3, spacing=spacing)

    def get_gaussian(self, cov, alpha_cut=3):

        if cov.shape != (self._dimension, self._dimension):
            raise ValueError("Numpy array 'cov' must be of shape (%d,%d)" %
                             (self._dimension, self._dimension))

        # Generate intervals for x, y and z based on cut-off distance given by
        # standard deviation, alpha_cut and spacing
        [x_max, y_max, z_max] = np.ceil(
            np.sqrt(cov.diagonal()) * alpha_cut/self._spacing)

        step = 1
        x_interval = np.arange(-x_max, x_max+step, step)
        y_interval = np.arange(-y_max, y_max+step, step)
        z_interval = np.arange(-z_max, z_max+step, step)

        # Generate arrays of 3D points bearing in mind that nifti-nda.shape =
        # (z,y,x)-coord. 'ij' yields vertical x-coordinate for image!
        [X, Y, Z] = np.meshgrid(x_interval, y_interval, z_interval,
                                indexing='ij')
        points = np.array([Z.flatten(), Y.flatten(), X.flatten()])

        # Mean/Origin for Gaussian blurring
        origin = np.zeros((self._dimension, 1))

        # Scaling matrix depending on spacing
        S = np.diag(self._spacing)

        # Compute scaled, inverse covariance matrix
        cov_scale_inv = S.dot(np.linalg.inv(cov)).dot(S)

        # Compute Gaussian weights
        values = np.sum((points-origin)*cov_scale_inv.dot(points-origin), 0)
        kernel = np.exp(-0.5*values)
        kernel = kernel/np.sum(kernel)

        # Reshape kernel
        kernel = kernel.reshape(
            x_interval.size, y_interval.size, z_interval.size)

        return kernel

    def get_dx_forward_difference(self):

        # kernel = np.zeros((z,y,x))
        kernel = np.zeros((1, 1, 2))
        kernel[:] = np.array([1, -1])

        return kernel / self._spacing[0]

    def get_dx_backward_difference(self):

        # kernel = np.zeros((z,y,x))
        kernel = np.zeros((1, 1, 3))
        kernel[:] = np.array([0, 1, -1])

        return kernel / self._spacing[0]

    def get_dy_forward_difference(self):

        # kernel = np.zeros((z,y,x))
        kernel = np.zeros((1, 2, 1))
        kernel[:] = np.array([[1], [-1]])

        return kernel / self._spacing[1]

    def get_dy_backward_difference(self):

        # kernel = np.zeros((z,y,x))
        kernel = np.zeros((1, 3, 1))
        kernel[:] = np.array([[0], [1], [-1]])

        return kernel / self._spacing[1]

    def get_dz_forward_difference(self):

        # kernel = np.zeros((z,y,x))
        kernel = np.zeros((2, 1, 1))
        kernel[:] = np.array([[[1]], [[-1]]])

        return kernel / self._spacing[2]

    def get_dz_backward_difference(self):

        # kernel = np.zeros((z,y,x))
        kernel = np.zeros((3, 1, 1))
        kernel[:] = np.array([[[0]], [[1]], [[-1]]])

        return kernel / self._spacing[2]
