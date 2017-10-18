##
# \file solvers_test.py
#  \brief  Class containing unit tests for numerical solvers
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2017


# Import libraries
import os
import numpy as np
import unittest
import sys
import matplotlib.pyplot as plt
import pysitk.python_helper as ph

# Import modules
import nsol.linear_operators as LinearOperators
import nsol.noise as Noise
import nsol.tikhonov_linear_solver as tk
import nsol.admm_linear_solver as admm
import nsol.primal_dual_solver as pd
import nsol.observer as observer
from nsol.proximal_operators import ProximalOperators as prox
from nsol.similarity_measures import SimilarityMeasures as sim_meas

from nsol.definitions import DIR_TEST


class SolversTest(unittest.TestCase):

    def _get_operators(self, x_gt, A, A_adj, grad, grad_adj):

        # A: X \rightarrow Y and grad: X \rightarrow Z
        X_shape = x_gt.shape
        Y_shape = A(x_gt).shape
        Z_shape = grad(x_gt).shape

        I_ = lambda x: x.flatten()
        I_adj_ = lambda x: x.flatten()

        A_ = lambda x: A(x.reshape(*X_shape)).flatten()
        A_adj_ = lambda x: A_adj(x.reshape(*Y_shape)).flatten()

        D_ = lambda x: grad(x.reshape(*X_shape)).flatten()
        D_adj_ = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

        return I_, I_adj_, A_, A_adj_, D_, D_adj_

    def setUp(self):
        self.accuracy = 7

        # Noise level for corrupting image
        self.noise_level = 0.05

        # Standard deviation for blurring
        sigma2 = 1.5

        self.ssd = lambda x: sim_meas.sum_of_squared_differences(x, x_ref)
        self.mse = lambda x: sim_meas.mean_squared_error(x, x_ref)
        self.rmse = lambda x: sim_meas.root_mean_square_error(x, x_ref)
        self.psnr = lambda x: sim_meas.peak_signal_to_noise_ratio(x, x_ref)
        self.ssim = lambda x: sim_meas.structural_similarity(x, x_ref)
        self.ncc = lambda x: sim_meas.normalized_cross_correlation(x, x_ref)
        self.mi = lambda x: sim_meas.mutual_information(x, x_ref)
        self.nmi = lambda x: sim_meas.normalized_mutual_information(x, x_ref)

        # ---------------------------------1D---------------------------------

        # Ground-truth solution
        self.x_gt_1D = np.ones(50) * 50
        self.x_gt_1D[5] = 10
        self.x_gt_1D[16] = 100
        self.x_gt_1D[23] = 150
        self.x_gt_1D[30] = 20

        linear_operators = LinearOperators.LinearOperators1D()
        self.A_1D, self.A_adj_1D = linear_operators.\
            get_gaussian_blurring_operators(sigma2)
        self.grad_1D, self.grad_adj_1D = linear_operators.\
            get_gradient_operators()

        # ---------------------------------2D---------------------------------

        # filename_2D = os.path.join(DIR_TEST, "2D_Lena_256.png")
        filename_2D = os.path.join(DIR_TEST, "2D_BrainWeb.png")

        # Ground-truth solution
        self.x_gt_2D = ph.read_image(filename_2D).astype(np.float64)

        cov = np.diag(np.ones(2)) * sigma2
        linear_operators = LinearOperators.LinearOperators2D()
        self.A_2D, self.A_adj_2D = linear_operators.\
            get_gaussian_blurring_operators(cov)
        self.grad_2D, self.grad_adj_2D = linear_operators.\
            get_gradient_operators()

    ##
    # Test x_scale setting for solver in the context of deblurring in 1D
    # \date       2017-08-04 16:46:07+0100
    #
    def test_x_scale_1D(self):

        x_scale = self.x_gt_1D.max()

        I, I_adj, A, A_adj, D, D_adj = self._get_operators(
            self.x_gt_1D, self.A_1D, self.A_adj_1D,
            self.grad_1D, self.grad_adj_1D)

        # -----------------------------Scale Data-----------------------------

        x_ = self.x_gt_1D / x_scale

        # Blur
        b_ = A(x_)

        # Add noise
        noise = Noise.Noise(b_, seed=1)
        noise.add_poisson_noise(noise_level=self.noise_level)
        b_ = noise.get_noisy_data()

        b = b_.flatten()
        x0 = np.array(b)

        solver = tk.TikhonovLinearSolver(
            A=A, A_adj=A_adj,
            B=D, B_adj=D_adj,
            b=b,
            x0=x0,
            x_scale=1,
        )
        solver.run()
        recon_tk1 = solver.get_x().reshape(*self.x_gt_1D.shape)

        solver = admm.ADMMLinearSolver(
            A=A, A_adj=A_adj,
            B=D, B_adj=D_adj,
            b=b,
            x0=x0,
            x_scale=1,
            dimension=1,
        )
        solver.run()
        recon_admm1 = solver.get_x().reshape(*self.x_gt_1D.shape)

        solver = pd.PrimalDualSolver(
            prox_f=lambda x, tau: prox.prox_linear_least_squares(
                x=x, tau=tau,
                A=A, A_adj=A_adj, b=b, x0=x0, x_scale=1),
            prox_g_conj=prox.prox_tv_conj,
            B=D,
            B_conj=D_adj,
            L2=8,
            x0=x0,
            x_scale=1,
        )
        solver.run()
        recon_pd1 = solver.get_x().reshape(*self.x_gt_1D.shape)

        # ----------------------------Solver Scale----------------------------

        x_ = self.x_gt_1D

        # Blur
        b_ = A(x_)

        # Add noise
        noise = Noise.Noise(b_, seed=1)
        noise.add_poisson_noise(noise_level=self.noise_level)
        b_ = noise.get_noisy_data()

        b = b_.flatten()
        x0 = np.array(b)

        solver = tk.TikhonovLinearSolver(
            A=A, A_adj=A_adj,
            B=D, B_adj=D_adj,
            b=b,
            x0=x0,
            x_scale=x_scale,
        )
        solver.run()
        recon_tk2 = solver.get_x().reshape(*self.x_gt_1D.shape)

        solver = admm.ADMMLinearSolver(
            A=A, A_adj=A_adj,
            B=D, B_adj=D_adj,
            b=b,
            x0=x0,
            x_scale=x_scale,
            dimension=1,
        )
        solver.run()
        recon_admm2 = solver.get_x().reshape(*self.x_gt_1D.shape)

        solver = pd.PrimalDualSolver(
            prox_f=lambda x, tau: prox.prox_linear_least_squares(
                x=x, tau=tau,
                A=A, A_adj=A_adj, b=b, x0=x0, x_scale=x_scale),
            prox_g_conj=prox.prox_tv_conj,
            B=D,
            B_conj=D_adj,
            L2=8,
            x0=x0,
            x_scale=x_scale,
        )
        solver.run()
        recon_pd2 = solver.get_x().reshape(*self.x_gt_1D.shape)

        # ----------------------------Check results----------------------------
        # Check Tikhonov
        self.assertEqual(np.round(
            np.linalg.norm(recon_tk2 - recon_tk1 * x_scale),
            decimals=self.accuracy), 0)

        # Check ADMM
        self.assertEqual(np.round(
            np.linalg.norm(recon_admm2 - recon_admm1 * x_scale),
            decimals=self.accuracy), 0)

        # Check Primal-Dual
        self.assertEqual(np.round(
            np.linalg.norm(recon_pd2 - recon_pd1 * x_scale),
            decimals=self.accuracy), 0)

    ##
    # Test x_scale setting for solver in the context of deblurring in 2D
    # \date       2017-08-04 16:46:07+0100
    #
    def test_x_scale_2D(self):

        x_scale = self.x_gt_2D.max()

        I, I_adj, A, A_adj, D, D_adj = self._get_operators(
            self.x_gt_2D, self.A_2D, self.A_adj_2D,
            self.grad_2D, self.grad_adj_2D)

        # -----------------------------Scale Data-----------------------------

        x_ = self.x_gt_2D / x_scale

        # Blur
        b_ = A(x_)

        # Add noise
        noise = Noise.Noise(b_, seed=1)
        noise.add_poisson_noise(noise_level=self.noise_level)
        b_ = noise.get_noisy_data()

        b = b_.flatten()
        x0 = np.array(b)

        solver = tk.TikhonovLinearSolver(
            A=A, A_adj=A_adj,
            B=D, B_adj=D_adj,
            b=b,
            x0=x0,
            x_scale=1,
        )
        solver.run()
        recon_tk1 = solver.get_x().reshape(*self.x_gt_2D.shape)

        solver = admm.ADMMLinearSolver(
            A=A, A_adj=A_adj,
            B=D, B_adj=D_adj,
            b=b,
            x0=x0,
            x_scale=1,
            dimension=2,
        )
        solver.run()
        recon_admm1 = solver.get_x().reshape(*self.x_gt_2D.shape)

        solver = pd.PrimalDualSolver(
            prox_f=lambda x, tau: prox.prox_linear_least_squares(
                x=x, tau=tau,
                A=A, A_adj=A_adj, b=b, x0=x0, x_scale=1),
            prox_g_conj=prox.prox_tv_conj,
            B=D,
            B_conj=D_adj,
            L2=8,
            x0=x0,
            x_scale=1,
        )
        solver.run()
        recon_pd1 = solver.get_x().reshape(*self.x_gt_2D.shape)

        # ----------------------------Solver Scale----------------------------

        x_ = self.x_gt_2D

        # Blur
        b_ = A(x_)

        # Add noise
        noise = Noise.Noise(b_, seed=1)
        noise.add_poisson_noise(noise_level=self.noise_level)
        b_ = noise.get_noisy_data()

        b = b_.flatten()
        x0 = np.array(b)

        solver = tk.TikhonovLinearSolver(
            A=A, A_adj=A_adj,
            B=D, B_adj=D_adj,
            b=b,
            x0=x0,
            x_scale=x_scale,
        )
        solver.run()
        recon_tk2 = solver.get_x().reshape(*self.x_gt_2D.shape)

        solver = admm.ADMMLinearSolver(
            A=A, A_adj=A_adj,
            B=D, B_adj=D_adj,
            b=b,
            x0=x0,
            x_scale=x_scale,
            dimension=2,
        )
        solver.run()
        recon_admm2 = solver.get_x().reshape(*self.x_gt_2D.shape)

        solver = pd.PrimalDualSolver(
            prox_f=lambda x, tau: prox.prox_linear_least_squares(
                x=x, tau=tau,
                A=A, A_adj=A_adj, b=b, x0=x0, x_scale=x_scale),
            prox_g_conj=prox.prox_tv_conj,
            B=D,
            B_conj=D_adj,
            L2=8,
            x0=x0,
            x_scale=x_scale,
        )
        solver.run()
        recon_pd2 = solver.get_x().reshape(*self.x_gt_2D.shape)

        # ----------------------------Check results----------------------------
        # Check Tikhonov
        self.assertEqual(np.round(
            np.linalg.norm(recon_tk2 - recon_tk1 * x_scale),
            decimals=self.accuracy), 0)

        # Check ADMM
        self.assertEqual(np.round(
            np.linalg.norm(recon_admm2 - recon_admm1 * x_scale),
            decimals=self.accuracy), 0)

        # Check Primal-Dual
        self.assertEqual(np.round(
            np.linalg.norm(recon_pd2 - recon_pd1 * x_scale),
            decimals=self.accuracy), 0)
