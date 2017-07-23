##
# \file TestLossFunctions.py
#  \brief  Class containing unit tests for lossFunctions
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


# Import libraries
import numpy as np
import unittest
import sys
import matplotlib.pyplot as plt
import pythonhelper.PythonHelper as ph

# Import modules
import numericalsolver.lossFunctions as lf



class TestLossFunctions(unittest.TestCase):

    accuracy = 7
    m = 500     # 4e5
    n = 1000    # 1e6

    def setUp(self):
        self.accuracy = 7
        self.m = 500
        self.n = 1000

        self.A = np.random.rand(self.m, self.n)
        self.b = np.random.rand(self.m) * 10

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

        for i in xrange(0, self.m):
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

        for i in xrange(0, self.m):
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

        for i in xrange(0, self.m):
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

        M = 20
        steps = 100*M
        residual = np.linspace(-M, M, steps)

        residual2 = residual**2

        losses = ["linear", "soft_l1", "huber"]
        loss = []
        grad_loss = []
        jac = []
        labels = []

        loss.append(lf.linear(residual2))
        grad_loss.append(lf.gradient_linear(residual2))
        jac.append(lf.gradient_linear(residual2)*residual)
        labels.append("linear")

        loss.append(lf.soft_l1(residual2))
        grad_loss.append(lf.gradient_soft_l1(residual2))
        jac.append(lf.gradient_soft_l1(residual2)*residual)
        labels.append("soft_l1")

        for gamma in (1, 1.345, 5, 10, 15):
            loss.append(lf.huber(residual2, gamma=gamma))
            grad_loss.append(lf.gradient_huber(residual2, gamma=gamma))
            jac.append(lf.gradient_huber(residual2, gamma=gamma)*residual)
            labels.append("huber(" + str(gamma) + ")")

        ph.show_curves(loss, x=residual, labels=labels,
                       title="losses rho(x^2)")
        ph.show_curves(grad_loss, x=residual, labels=labels,
                       title="gradient losses rho'(x^2)")
        ph.show_curves(jac, x=residual, labels=labels,
                       title="jacobian rho'(x^2)*x")
