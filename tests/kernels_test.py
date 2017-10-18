##
# \file kernels_test.py
#  \brief  Class containing unit tests for module Kernels
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date July 2017

import numpy as np
import scipy
import unittest

import nsol.kernels as Kernels
import nsol.linear_operators as LinearOperators

from nsol.definitions import DIR_TEST


class KernelsTest(unittest.TestCase):

    def setUp(self):

        self.accuracy = 10
        self.scaling = 1

        # 1D
        self.shape_1D = (50,)

        self.cov_1D = 2

        self.x_1D = self.scaling * np.random.rand(*self.shape_1D)
        self.y_1D = self.scaling * np.random.rand(*self.shape_1D)

        self.linear_operators_1D = LinearOperators.LinearOperators1D()

        # 2D
        self.shape_2D = (50, 50)

        self.cov_2D = np.zeros((2, 2))
        self.cov_2D[0, 0] = 2
        self.cov_2D[1, 1] = 2

        self.x_2D = self.scaling * np.random.rand(*self.shape_2D)
        self.y_2D = self.scaling * np.random.rand(*self.shape_2D)

        self.linear_operators_2D = LinearOperators.LinearOperators2D()

        # 3D
        self.shape_3D = (50, 50, 10)
        self.cov_3D = np.zeros((3, 3))
        self.cov_3D[0, 0] = 2
        self.cov_3D[1, 1] = 2
        self.cov_3D[2, 2] = 2

        self.x_3D = self.scaling * np.random.rand(*self.shape_3D)
        self.y_3D = self.scaling * np.random.rand(*self.shape_3D)

        self.linear_operators_3D = LinearOperators.LinearOperators3D()

    def test_kernel_spacings(self):
        # ---------------------------------1D----------------------------------
        spacing = 2
        kernel = Kernels.Kernels1D(spacing=spacing)

        # dx
        k = kernel.get_dx_forward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing)
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        k = kernel.get_dx_backward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing)
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # ---------------------------------2D----------------------------------
        spacing = np.array([2, 4])
        kernel = Kernels.Kernels2D(spacing=spacing)

        # dx
        k = kernel.get_dx_forward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing[0])
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        k = kernel.get_dx_backward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing[0])
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # dy
        k = kernel.get_dy_forward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing[1])
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        k = kernel.get_dy_backward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing[1])
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # ---------------------------------3D----------------------------------
        spacing = np.array([2, 4, 6])
        kernel = Kernels.Kernels3D(spacing=spacing)

        # dx
        k = kernel.get_dx_forward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing[0])
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        k = kernel.get_dx_backward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing[0])
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # dy
        k = kernel.get_dy_forward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing[1])
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        k = kernel.get_dy_backward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing[1])
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # dz
        k = kernel.get_dz_forward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing[2])
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        k = kernel.get_dz_backward_difference()
        abs_diff = np.linalg.norm(
            np.abs(k[k.nonzero()]) - 1./spacing[2])
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

    def test_gaussian_blurring(self):

        # ---------------------------------1D----------------------------------
        A, A_adj = self.linear_operators_1D.get_gaussian_blurring_operators(
            self.cov_1D)

        A_x = A(self.x_1D)
        A_adj_y = A_adj(self.y_1D)

        # Check |(Ax,y) - (x,A'y)| = 0
        abs_diff = np.abs(np.sum(A_x*self.y_1D) - np.sum(self.x_1D*A_adj_y))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # ---------------------------------2D----------------------------------
        A, A_adj = self.linear_operators_2D.get_gaussian_blurring_operators(
            self.cov_2D)

        A_x = A(self.x_2D)
        A_adj_y = A_adj(self.y_2D)

        # Check |(Ax,y) - (x,A'y)| = 0
        abs_diff = np.abs(np.sum(A_x*self.y_2D) - np.sum(self.x_2D*A_adj_y))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # ---------------------------------3D----------------------------------
        A, A_adj = self.linear_operators_3D.get_gaussian_blurring_operators(
            self.cov_3D)
        A_x = A(self.x_3D)
        A_adj_y = A_adj(self.y_3D)

        # Check |(Ax,y) - (x,A'y)| = 0
        abs_diff = np.abs(np.sum(A_x*self.y_3D) - np.sum(self.x_3D*A_adj_y))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

    def test_differential_operations(self):

        # ---------------------------------1D----------------------------------

        # dx
        D, D_adj = self.linear_operators_1D.get_dx_operators()
        D_x = D(self.x_1D)
        D_adj_y = D_adj(self.y_1D)
        abs_diff = np.abs(np.sum(D_x*self.y_1D) - np.sum(self.x_1D*D_adj_y))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # ---------------------------------2D----------------------------------

        # dx
        D, D_adj = self.linear_operators_2D.get_dx_operators()
        D_x = D(self.x_2D)
        D_adj_y = D_adj(self.y_2D)
        abs_diff = np.abs(np.sum(D_x*self.y_2D) - np.sum(self.x_2D*D_adj_y))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # dy
        D, D_adj = self.linear_operators_2D.get_dy_operators()
        D_x = D(self.x_2D)
        D_adj_y = D_adj(self.y_2D)
        abs_diff = np.abs(np.sum(D_x*self.y_2D) - np.sum(self.x_2D*D_adj_y))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # ---------------------------------3D----------------------------------

        # dx
        D, D_adj = self.linear_operators_3D.get_dx_operators()
        D_x = D(self.x_3D)
        D_adj_y = D_adj(self.y_3D)
        abs_diff = np.abs(np.sum(D_x*self.y_3D) - np.sum(self.x_3D*D_adj_y))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # dy
        D, D_adj = self.linear_operators_3D.get_dy_operators()
        D_x = D(self.x_3D)
        D_adj_y = D_adj(self.y_3D)
        abs_diff = np.abs(np.sum(D_x*self.y_3D) - np.sum(self.x_3D*D_adj_y))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # dz
        D, D_adj = self.linear_operators_3D.get_dz_operators()
        D_x = D(self.x_3D)
        D_adj_y = D_adj(self.y_3D)
        abs_diff = np.abs(np.sum(D_x*self.y_3D) - np.sum(self.x_3D*D_adj_y))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

    def test_get_gradient_operators(self):

        # ---------------------------------1D----------------------------------
        dx, dx_adj = self.linear_operators_1D.get_dx_operators()
        grad, grad_adj = self.linear_operators_1D.get_gradient_operators()

        # grad
        x = self.x_1D
        y0 = dx(x)
        res = y0
        res_ = grad(x)
        abs_diff = np.abs(np.sum(res_-res))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # grad_adj
        y = self.scaling * np.random.rand(*self.shape_1D)

        y0 = y
        res = dx_adj(y0)
        res_ = grad_adj(y)
        abs_diff = np.abs(np.sum(res_-res))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # ---------------------------------2D----------------------------------
        dx, dx_adj = self.linear_operators_2D.get_dx_operators()
        dy, dy_adj = self.linear_operators_2D.get_dy_operators()
        grad, grad_adj = self.linear_operators_2D.get_gradient_operators()

        # grad
        x = self.x_2D
        y0 = dx(x)
        y1 = dy(x)
        res = np.concatenate((y0, y1))
        res_ = grad(x)
        abs_diff = np.abs(np.sum(res_-res))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # grad_adj
        y = self.scaling * np.concatenate((
            np.random.rand(*self.shape_2D),
            np.random.rand(*self.shape_2D)))

        m = self.shape_2D[0]
        y0 = y[:m, :]
        y1 = y[m:2*m, :]
        res = dx_adj(y0) + dy_adj(y1)
        res_ = grad_adj(y)
        abs_diff = np.abs(np.sum(res_-res))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # ---------------------------------3D----------------------------------
        dx, dx_adj = self.linear_operators_3D.get_dx_operators()
        dy, dy_adj = self.linear_operators_3D.get_dy_operators()
        dz, dz_adj = self.linear_operators_3D.get_dz_operators()
        grad, grad_adj = self.linear_operators_3D.get_gradient_operators()

        # grad
        x = self.x_3D
        y0 = dx(x)
        y1 = dy(x)
        y2 = dz(x)
        res = np.concatenate((y0, y1, y2))
        res_ = grad(x)
        abs_diff = np.abs(np.sum(res_-res))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # grad_adj
        y = self.scaling * np.concatenate((
            0*np.random.rand(*self.shape_3D),
            0*np.random.rand(*self.shape_3D),
            np.random.rand(*self.shape_3D)))

        m = self.shape_3D[0]
        y0 = y[:m, ...]
        y1 = y[m:2*m, ...]
        y2 = y[2*m:, ...]
        res = dx_adj(y0) + dy_adj(y1) + dz_adj(y2)
        res_ = grad_adj(y)
        abs_diff = np.abs(np.sum(res_-res))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

    def test_gradient_operators(self):

        # ---------------------------------1D-------------------------------

        grad, grad_adj = self.linear_operators_1D.get_gradient_operators()
        x = self.x_1D
        y = self.scaling * np.random.rand(*self.shape_1D)

        abs_diff = np.abs(np.sum(grad(x)*y) - np.sum(x*grad_adj(y)))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # ---------------------------------2D-------------------------------

        grad, grad_adj = self.linear_operators_2D.get_gradient_operators()
        x = self.x_2D
        y = self.scaling * np.concatenate((
            np.random.rand(*self.shape_2D),
            np.random.rand(*self.shape_2D)))

        abs_diff = np.abs(np.sum(grad(x)*y) - np.sum(x*grad_adj(y)))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)

        # ---------------------------------3D-------------------------------

        grad, grad_adj = self.linear_operators_3D.get_gradient_operators()
        x = self.x_3D
        y = self.scaling * np.concatenate((
            np.random.rand(*self.shape_3D),
            np.random.rand(*self.shape_3D),
            np.random.rand(*self.shape_3D)))

        abs_diff = np.abs(np.sum(grad(x)*y) - np.sum(x*grad_adj(y)))
        self.assertEqual(np.round(abs_diff, decimals=self.accuracy), 0)
