##
# \file loss_functions_test.py
#  \brief  Class containing unit tests for LossFunctions
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2017


# Import libraries
import numpy as np
import unittest
import sys
import pysitk.python_helper as ph

# Import modules
from nsol.loss_functions import LossFunctions as lf


class LossFunctionsTest(unittest.TestCase):

    def setUp(self):

        self.accuracy = 7
        self.m = 500
        self.n = 1000

        self.A = np.random.rand(self.m, self.n)
        self.b = np.random.rand(self.m) * 10
        self.x = np.random.rand(self.n) * 23

    # -----------------------------Loss functions-----------------------------

    def test_linear(self):

        diff = lf.linear(self.b) - self.b

        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        diff = lf.gradient_linear(self.b) - 1
        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

    def test_soft_l1(self):

        soft_l1 = np.zeros_like(self.b)
        soft_l1_grad = np.zeros_like(self.b)

        for i in range(0, self.m):
            e = self.b[i]
            soft_l1[i] = 2*(np.sqrt(1+e)-1)
            soft_l1_grad[i] = 1./np.sqrt(1+e)

        diff = lf.soft_l1(self.b) - soft_l1

        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        diff_grad = lf.gradient_soft_l1(self.b) - soft_l1_grad
        self.assertEqual(np.around(
            np.linalg.norm(diff_grad), decimals=self.accuracy), 0)

    def test_huber(self):

        gamma = 1.3
        gamma2 = gamma * gamma

        huber = np.zeros_like(self.b)
        grad_huber = np.zeros_like(self.b)

        for i in range(0, self.m):
            e = self.b[i]
            if e < gamma2:
                huber[i] = e
                grad_huber[i] = 1
            else:
                huber[i] = 2*gamma*np.sqrt(e) - gamma2
                grad_huber[i] = gamma/np.sqrt(e)

        diff = lf.huber(self.b, gamma=gamma) - huber

        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        diff_grad = lf.gradient_huber(self.b, gamma=gamma) - grad_huber
        self.assertEqual(np.around(
            np.linalg.norm(diff_grad), decimals=self.accuracy), 0)

    def test_cauchy(self):

        cauchy = np.zeros_like(self.b)
        cauchy_grad = np.zeros_like(self.b)

        for i in range(0, self.m):
            e = self.b[i]
            cauchy[i] = np.log(1+e)
            cauchy_grad[i] = 1./(1+e)

        diff = lf.cauchy(self.b) - cauchy

        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        diff_grad = lf.gradient_cauchy(self.b) - cauchy_grad
        self.assertEqual(np.around(
            np.linalg.norm(diff_grad), decimals=self.accuracy), 0)

    def test_arctan(self):

        arctan = np.zeros_like(self.b)
        arctan_grad = np.zeros_like(self.b)

        for i in range(0, self.m):
            e = self.b[i]
            arctan[i] = np.arctan(e)
            arctan_grad[i] = 1. / (1 + e*e)

        diff = lf.arctan(self.b) - arctan

        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        diff_grad = lf.gradient_arctan(self.b) - arctan_grad
        self.assertEqual(np.around(
            np.linalg.norm(diff_grad), decimals=self.accuracy), 0)

    def test_show_curves(self):

        flag_show_curves = 0
        if flag_show_curves:
            M = 50
            steps = 100 * M
            residual = np.linspace(-M, M, steps)

            residual2 = np.array(residual**2)

            losses = []
            grad_losses = []
            jacobians = []
            labels = []

            for loss in ["linear", "soft_l1", "cauchy", "arctan"]:
                for i, f_scale in enumerate([1., 1.5]):
                    if loss == "linear" and i > 0:
                        continue
                    # print loss, f_scale
                    label = loss
                    if loss != "linear":
                        label += "_scale"+str(f_scale)
                    losses.append(np.array(lf.get_loss[loss](
                        f2=residual2, f_scale=f_scale)))
                    grad_losses.append(np.array(lf.get_gradient_loss[loss](
                        f2=residual2, f_scale=f_scale)))
                    jacobians.append(np.array(lf.get_gradient_loss[loss](
                        f2=residual2, f_scale=f_scale)*residual))
                    labels.append(label)

            # losses.append(lf.soft_l1(residual2))
            # grad_losses.append(lf.gradient_soft_l1(residual2))
            # jacobians.append(lf.gradient_soft_l1(residual2)*residual)
            # labels.append("soft_l1")

            for gamma in [1, 1.5]:
                losses.append(np.array(lf.huber(residual2, gamma=gamma)))
                grad_losses.append(
                    np.array(lf.gradient_huber(residual2, gamma=gamma)))
                jacobians.append(np.array(lf.gradient_huber(
                    residual2, gamma=gamma)*residual))
                labels.append("huber(" + str(gamma) + ")")

            x = residual
            ph.show_curves(losses, x=x, labels=labels, xlabel="x",
                           title="Cost rho(x^2)")
            ph.show_curves(grad_losses, x=x, labels=labels, xlabel="x",
                           title="Gradient Loss rho'(x^2)")
            ph.show_curves(jacobians, x=x, labels=labels, xlabel="x",
                           title="Gradient Cost rho'(x^2)*x")

    # --------------------Conversion from residual to cost--------------------

    def test_cost_from_residual_linear(self):

        loss = "linear"

        def f(x):
            nda = np.zeros(3)
            nda[0] = x[0]**2 - 3*x[1]**3 + 5
            nda[1] = 2*x[0] + x[1]**2 - 1
            nda[2] = x[0] + x[1]
            return nda

        def df(x):
            nda = np.zeros((3, 2))
            nda[0, 0] = 2*x[0]
            nda[1, 0] = 2
            nda[2, 0] = 1

            nda[0, 1] = -9*x[1]**2
            nda[1, 1] = 2*x[1]
            nda[2, 1] = 1
            return nda

        X0, X1 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(0, 10, 0.2))
        points = np.array([X1.flatten(), X0.flatten()])

        cost_gd = lambda x: 0.5 * np.sum(f(x)**2)
        grad_cost_gd = lambda x: np.sum(f(x)[:, np.newaxis]*df(x), 0)

        for i in range(points.shape[1]):
            point = points[:, i]
            diff_cost = cost_gd(point) - \
                lf.get_ell2_cost_from_residual(f(point), loss=loss)
            diff_grad = grad_cost_gd(point) - \
                lf.get_gradient_ell2_cost_from_residual(
                    f(point), df(point), loss=loss)

            self.assertEqual(np.around(
                np.linalg.norm(diff_cost), decimals=self.accuracy), 0)
            self.assertEqual(np.around(
                np.linalg.norm(diff_grad), decimals=self.accuracy), 0)

    def test_linear_least_squares_cost_gradient(self):

        residual = self.A.dot(self.x) - self.b
        ell2 = 0.5*np.sum(residual**2)
        diff = lf.get_ell2_cost_from_residual(f=residual, loss="linear")
        diff -= ell2
        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        grad_ell2 = self.A.transpose().dot(residual)
        diff = lf.get_gradient_ell2_cost_from_residual(
            f=residual,
            jac_f=self.A,
            loss="linear")
        diff -= grad_ell2
        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

    def test_soft_l1_least_squares_cost_gradient(self):

        residual = self.A.dot(self.x) - self.b
        ell2 = 0.5*np.sum(lf.soft_l1(residual**2))
        diff = lf.get_ell2_cost_from_residual(f=residual, loss="soft_l1")
        diff -= ell2
        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        # Derive analytical cost function; However, unit tests in
        # registration_test.py suggest that all good
    #     grad_ell2 = lf.gradient_soft_l1(res**2).dot(self.A.transpose().dot(residual))
    #     diff = lf.get_gradient_ell2_cost_from_residual(
    #         f=residual,
    #         jac_f=self.A,
    #         loss="linear")
    #     diff -= grad_ell2
    #     self.assertEqual(np.around(
    #         np.linalg.norm(diff), decimals=self.accuracy), 0)
