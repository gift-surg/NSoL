##
# \file linear_operators.py
# \brief      Class to create linear operators used for blurring and
#             differentiation in both 2D and 3D
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

# Import libraries
import numpy as np
import scipy.ndimage
from abc import ABCMeta, abstractmethod

import nsol.kernels as Kernels


class LinearOperators(object):
    __metaclass__ = ABCMeta

    ##
    # { constructor_description }
    # \date       2017-07-23 16:42:57+0100
    #
    # \param      self       The object
    # \param      dimension  Dimension of space, integer
    # \param      spacing    Spacing in each spatial direction in x- (y-, z-)
    #                        direction as numpy array or scalar (only 1D)
    #
    def __init__(self, dimension, spacing):

        self._dimension = dimension
        self._spacing = spacing

        if dimension == 1:
            self._kernels = Kernels.Kernels1D(spacing=spacing)
        elif dimension == 2:
            self._kernels = Kernels.Kernels2D(spacing=spacing)
        elif dimension == 3:
            self._kernels = Kernels.Kernels3D(spacing=spacing)

    def get_spacing(self):
        return self._spacing

    def get_dimension(self):
        return self._dimension

    ##
    # Gets the convolution and adjoint convolution operators given the provided
    # kernel
    # \date       2017-07-19 16:14:44+0100
    #
    # \param      self    The object
    # \param      kernel  Kernel specifying the linear operation as numpy array
    # \param      mode    Mode specifying the boundary conditions for the
    #                     convolution
    #
    # \return     The convolution and adjoint convolution operators.
    #
    def get_convolution_and_adjoint_convolution_operators(
            self, kernel, mode="wrap"):

        kernel_adj = kernel

        A = lambda x: scipy.ndimage.convolve(x, kernel, mode=mode)
        A_adj = lambda x: scipy.ndimage.convolve(x, kernel_adj, mode=mode)

        return A, A_adj

    ##
    # Gets the Gaussian blurring operator and its adjoint associated to a
    # covariance matrix.
    # \date       2017-07-19 16:16:18+0100
    #
    # \param      self       The object
    # \param      cov        Variance covariance matrix as numpy array
    # \param      alpha_cut  Cut-off distance in integer, i.e. 3 means cutting
    #                        off a 3 sigma in each direction
    #
    # \return     The gaussian blurring operators.
    #
    def get_gaussian_blurring_operators(self, cov, alpha_cut=3):

        kernel = self._kernels.get_gaussian(cov=cov, alpha_cut=alpha_cut)

        return self.get_convolution_and_adjoint_convolution_operators(kernel)

    ##
    # Gets the differential operator in x-direction and its adjoint
    # \date       2017-07-19 16:31:55+0100
    #
    # \param      self  The object
    # \param      mode  Mode specifying the boundary conditions for the
    #                   convolution
    #
    # \return     The x-differential operators.
    #
    def get_dx_operators(self, mode="constant"):

        kernel = self._kernels.get_dx_forward_difference()
        kernel_adj = -self._kernels.get_dx_backward_difference()

        D = lambda x: scipy.ndimage.convolve(x, kernel, mode=mode)
        D_adj = lambda x: scipy.ndimage.convolve(x, kernel_adj, mode=mode)

        return D, D_adj

    ##
    # Gets the gradient operator and its adjoint for both 2D and 3D.
    #
    # Operator \p grad applied on (m x n) numpy array returns an (dim*m x n)
    # numpy array, i.e. stacking the differentials on top of each other.
    # Operator \p grad_adj maps from (dim*m x n) to (m x n)
    # \date       2017-07-19 17:14:26+0100
    #
    # \param      self  The object
    # \param      mode  The mode
    #
    # \return     The gradient operators.
    #
    def get_gradient_operators(self, mode="constant"):

        Dx, Dx_adj = self.get_dx_operators(mode=mode)

        if self._dimension == 1:
            grad = Dx
            grad_adj = Dx_adj

        if self._dimension == 2:
            Dy, Dy_adj = self.get_dy_operators(mode=mode)

            grad = lambda x: np.concatenate((Dx(x), Dy(x)))
            grad_adj = lambda x: self._get_adjoint_gradient_operator(
                x, [Dx_adj, Dy_adj])

        elif self._dimension == 3:
            Dy, Dy_adj = self.get_dy_operators(mode=mode)
            Dz, Dz_adj = self.get_dz_operators(mode=mode)

            grad = lambda x: np.concatenate((Dx(x), Dy(x), Dz(x)))
            grad_adj = lambda x: self._get_adjoint_gradient_operator(
                x, [Dx_adj, Dy_adj, Dz_adj])

        return grad, grad_adj

    ##
    # Gets the adjoint gradient operator.
    #
    # Apply Dx_adj(x[0:m,...]) + Dy_adj(x[m:2m,...]) (+ Dz_adj(x[2m:,...]))
    # \date       2017-07-19 17:19:06+0100
    #
    # \param      self        The object
    # \param      x           numpy array of shape (dim*m x n x p)
    # \param      D_adj_list  The d adj list
    #
    # \return     The adjoint gradient operator.
    #
    def _get_adjoint_gradient_operator(self, x, D_adj_list):

        x_split = np.array_split(x, self._dimension)

        D_adj_x_list = [D_adj_list[i](x_split[i])
                        for i in range(0, self._dimension)]

        D_adj_x = D_adj_x_list[0]
        for i in range(1, self._dimension):
            D_adj_x += D_adj_x_list[i]

        return D_adj_x


class LinearOperators1D(LinearOperators):

    def __init__(self, spacing=1):
        super(self.__class__, self).__init__(dimension=1, spacing=spacing)


class LinearOperators2D(LinearOperators):

    def __init__(self, spacing=np.ones(2)):
        super(self.__class__, self).__init__(dimension=2, spacing=spacing)

    ##
    # Gets the differential operator in y-direction and its adjoint
    # \date       2017-07-19 16:31:55+0100
    #
    # \param      self  The object
    # \param      mode  Mode specifying the boundary conditions for the
    #                   convolution
    #
    # \return     The y-differential operators.
    #
    def get_dy_operators(self, mode="constant"):

        kernel = self._kernels.get_dy_forward_difference()
        kernel_adj = -self._kernels.get_dy_backward_difference()

        D = lambda x: scipy.ndimage.convolve(x, kernel, mode=mode)
        D_adj = lambda x: scipy.ndimage.convolve(x, kernel_adj, mode=mode)

        return D, D_adj


class LinearOperators3D(LinearOperators):

    def __init__(self, spacing=np.ones(3)):
        super(self.__class__, self).__init__(dimension=3, spacing=spacing)

    ##
    # Gets the differential operator in y-direction and its adjoint
    # \date       2017-07-19 16:31:55+0100
    #
    # \param      self  The object
    # \param      mode  Mode specifying the boundary conditions for the
    #                   convolution
    #
    # \return     The y-differential operators.
    #
    def get_dy_operators(self, mode="constant"):

        kernel = self._kernels.get_dy_forward_difference()
        kernel_adj = -self._kernels.get_dy_backward_difference()

        D = lambda x: scipy.ndimage.convolve(x, kernel, mode=mode)
        D_adj = lambda x: scipy.ndimage.convolve(x, kernel_adj, mode=mode)

        return D, D_adj

    ##
    # Gets the differential operator in z-direction and its adjoint for 3D
    # \date       2017-07-19 16:31:55+0100
    #
    # \param      self  The object
    # \param      mode  Mode specifying the boundary conditions for the
    #                   convolution
    #
    # \return     The z-differential operators.
    #
    def get_dz_operators(self, mode="constant"):

        kernel = self._kernels.get_dz_forward_difference()
        kernel_adj = -self._kernels.get_dz_backward_difference()

        D = lambda x: scipy.ndimage.convolve(x, kernel, mode=mode)
        D_adj = lambda x: scipy.ndimage.convolve(x, kernel_adj, mode=mode)

        return D, D_adj
