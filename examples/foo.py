#!/usr/bin/python

##
# \file foo.py
# \brief      Playground file
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

import os
import sys
import numpy as np
import SimpleITK as sitk
import skimage.measure

import src.Kernels as Kernels
import src.LinearOperators as LinearOperators
import src.Noise as Noise
import src.TikhonovLinearSolver as tk
import src.ADMMLinearSolver as admm
import src.PrimalDualSolver as pd
import src.Monitor as monitor

sys.path.insert(1, os.path.join(
    os.path.abspath(os.environ['VOLUMETRIC_RECONSTRUCTION_DIR']), 'src', 'py'))
import utilities.PythonHelper as ph
import utilities.SimpleITKHelper as sitkh

from definitions import DIR_TEST


##
# Proximal operator for g(x) = 1/2 ||Ax-b||^2
# \date       2017-07-23 01:32:15+0100
#
# \param      x      { parameter_description }
# \param      tau    The tau
# \param      A      { parameter_description }
# \param      A_adj  The a adj
# \param      B      { parameter_description }
# \param      B_adj  The b adj
# \param      b      { parameter_description }
# \param      x0     The x 0
#
# \return     { description_of_the_return_value }
#
def prox_linear_least_squares(x, tau, A, A_adj, b, x0):
    # Identity
    I = lambda x: x

    tikhonov = tk.TikhonovLinearSolver(
        A=A, A_adj=A_adj, B=I, B_adj=I, b=b, x0=x0, b_reg=x, alpha=1./tau,
        verbose=0)
    tikhonov.run()
    return tikhonov.get_x()


##
# Proximal operator for ell1 denoising, i.e. g(x) = ||x - x0||_1.
#
# See Chambolle2011, p.135
# \date       2017-07-21 01:52:01+0100
#
# \param      x     numpy array
# \param      tau   step size; scalar >= 0
# \param      x0    observed; numpy array
#
# \return     prox_{\tau g}(x) as numpy array
#
def prox_ell1_denoising(x, tau, x0):
    return x0 + np.maximum(np.abs(x-x0) - tau, 0) * np.sign(x-x0)


##
# Proximal operator for least squares denoising, i.e. g(x) = 1/2 ||x - x0||_2^2
#
# With g(x) = 1/2 * ||x - x0||_2^2 it holds that
# prox_{\tau g}(x) = (x + \tau*x0) / (1 + \tau*x0).
# See Chambolle2011, p.133
# \date       2017-07-21 01:52:01+0100
#
# \param      x     numpy array
# \param      tau   step size; scalar >= 0
# \param      x0    observed; numpy array
#
# \return     prox_{\tau g}(x) as numpy array
#
def prox_ell2_denoising(x, tau, x0):
    return (x + tau * x0) / (1. + tau)


def prox_tv_conj(x, sigma):
    return x / np.maximum(1, np.abs(x))


def prox_huber_conj(x, sigma, alpha=0.05):
    x /= (1. + sigma * alpha)
    return x / np.maximum(1, np.abs(x))



np.random.seed(seed=1)

dimension = 1
dimension = 2
# dimension = 3

solver_TK = 1
solver_ADMM = 1
solver_PrimalDual = 1

flag_normalize = 1
flag_blurring = 0
flag_noise = 1
noise_level = 0.1

iter_max = 10
alpha = 0.03
rho = 1
ADMM_iterations = 30
PD_iterations = 30

cov = np.diag(np.ones(dimension)) * 1.5

if dimension == 1:
    cov = cov[0][0]
    original_nda = np.zeros(50)
    original_nda[5] = 10
    original_nda[16] = 100
    original_nda[23] = 150
    original_nda[30] = 50

elif dimension == 2:
    filename = os.path.join(DIR_TEST, "2D_Lena_256.png")
    # filename = os.path.join(DIR_TEST, "2D_Brain_Source.png")
    original_nda = ph.read_image(filename).astype(np.float64)

elif dimension == 3:
    filename = os.path.join(DIR_TEST, "3D_SheppLoganPhantom_64.nii.gz")
    original_sitk = sitk.ReadImage(filename)
    original_nda = sitk.GetArrayFromImage(original_sitk).astype(np.float64)

if flag_normalize:
    original_nda = original_nda / original_nda.max()

linear_operators = eval("LinearOperators.LinearOperators%dD()" % (dimension))

A, A_adj = linear_operators.get_gaussian_blurring_operators(cov)
grad, grad_adj = linear_operators.get_gradient_operators()

if flag_blurring:
    blurred_nda = A(original_nda)
else:
    blurred_nda = original_nda

noise = Noise.Noise(blurred_nda)
if flag_noise:
    # noise.add_gaussian_noise(noise_level=noise_level)
    noise.add_uniform_noise(noise_level=noise_level)
    # noise.add_salt_and_pepper_noise()
blurred_noisy_nda = noise.get_noisy_data()


# A: X \rightarrow Y and D: X \rightarrow Z
X_shape = original_nda.shape
Y_shape = A(original_nda).shape
Z_shape = grad(original_nda).shape

b = blurred_noisy_nda.flatten()
x0 = blurred_noisy_nda.flatten()

D = lambda x: grad(x)
D_adj = lambda x: grad_adj(x)

I_1D = lambda x: x
I_adj_1D = lambda x: x

A_1D = lambda x: A(x.reshape(*X_shape)).flatten()
A_adj_1D = lambda x: A_adj(x.reshape(*Y_shape)).flatten()

D_1D = lambda x: D(x.reshape(*X_shape)).flatten()
D_adj_1D = lambda x: D_adj(x.reshape(*Z_shape)).flatten()

prox_f = lambda x, tau: prox_linear_least_squares(
    x=x, tau=tau,
    A=A_1D, A_adj=A_adj_1D, b=b, x0=x0)

# prox_f = lambda x, tau: prox_ell1_denoising(x, tau, x0=b)
# prox_f = lambda x, tau: prox_ell2_denoising(x, tau, x0=b)

# prox_g_conj = lambda x, sigma: prox_tv_conj(
#     x=x, sigma=sigma)
# # prox_g_conj = lambda x, sigma: prox_huber_conj(
# #     x=x, sigma=sigma,
# #     alpha=0.05)

if not flag_blurring:
    A_1D = I_1D
    A_adj_1D = I_adj_1D

data_nda = [original_nda]
data_labels = ["original"]

# data_nda.append(blurred_nda)
# data_labels.append("blurred")

data_nda.append(blurred_noisy_nda)
data_labels.append("blurred+noise")

if dimension == 1:
    ph.show_curves(data_nda, labels=data_labels, fig_number=1)

elif dimension == 2:
    ph.show_arrays(data_nda, title=data_labels, fig_number=1)

elif dimension == 3:

    original_sitk = sitk.GetImageFromArray(original_nda)
    data_sitk = [original_sitk]

    # blurred_sitk = sitk.GetImageFromArray(blurred_nda)
    # data_sitk.append(blurred_sitk)

    blurred_noisy_sitk = sitk.GetImageFromArray(blurred_noisy_nda)
    data_sitk.append(blurred_noisy_sitk)

    ph.killall_itksnap()
    sitkh.show_sitk_image(data_sitk, label=data_labels)

x_ref = original_nda.flatten()
ssd = lambda x: np.sum(np.square(x-x_ref))
mse = lambda x: ssd(x) / x_ref.size
psnr = lambda x: 10 * np.log10(np.max(x)**2 / mse(x))
ssim = lambda x: skimage.measure.compare_ssim(x, x_ref)

monitors = []
measures_dic = {
    "SSD": ssd,
    # "MSE": mse,
    "PSNR": psnr,
    "SSIM": ssim,
}

if solver_TK:
    name = "TK"
    monitor_tk = monitor.Monitor(name)
    monitor_tk.set_measures(measures_dic)
    monitors.append(monitor_tk)
    solver = tk.TikhonovLinearSolver(
        A=A_1D, A_adj=A_adj_1D,
        B=D_1D, B_adj=D_adj_1D,
        b=b,
        alpha=alpha,
        x0=x0,
        # minimizer="lsmr",
        # minimizer="least_squares",
        # minimizer="L-BFGS-B",
        # data_loss="linear",
        # data_loss="huber",
        # data_loss="soft_l1",
        iter_max=iter_max,
    )
    solver.set_monitor(monitor_tk)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append(name)

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

if solver_ADMM:
    name = "ADMM-TV"
    monitor_admm = monitor.Monitor(name)
    monitor_admm.set_measures(measures_dic)
    monitors.append(monitor_admm)
    solver = admm.ADMMLinearSolver(
        A=A_1D, A_adj=A_adj_1D,
        b=b,
        B=D_1D, B_adj=D_adj_1D,
        x0=x0,
        alpha=alpha,
        rho=rho,
        ADMM_iterations=ADMM_iterations,
        dimension=dimension,
        iter_max=iter_max,
        # data_loss="huber",
    )
    solver.set_monitor(monitor_admm)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append(name)

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

if solver_PrimalDual:
    name = "PrimalDual-TV"
    monitor_pd = monitor.Monitor(name)
    monitor_pd.set_measures(measures_dic)
    monitors.append(monitor_pd)
    solver = pd.PrimalDualSolver(
        prox_f=prox_f,
        prox_g_conj=prox_tv_conj,
        B=D_1D,
        B_conj=D_adj_1D,
        L2=8,
        x0=x0,
        alpha=alpha,
        iterations=PD_iterations,
    )
    solver.set_monitor(monitor_pd)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append("PrimalDual-TV")

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

if solver_PrimalDual:
    name = "PrimalDual-Huber"
    monitor_pd = monitor.Monitor(name)
    monitor_pd.set_measures(measures_dic)
    monitors.append(monitor_pd)
    solver = pd.PrimalDualSolver(
        prox_f=prox_f,
        prox_g_conj=prox_huber_conj,
        B=D_1D,
        B_conj=D_adj_1D,
        L2=8,
        x0=x0,
        alpha=alpha,
        iterations=PD_iterations,
    )
    solver.set_monitor(monitor_pd)
    solver.run()
    recon_nda = solver.get_x().reshape(*X_shape)
    data_nda.append(recon_nda)
    data_labels.append(name)

    if dimension == 3:
        recon_sitk = sitk.GetImageFromArray(recon_nda)
        recon_sitk.CopyInformation(original_sitk)
        data_sitk.append(recon_sitk)

linestyles = ["-", ":", "-", "-.", "--"]
if dimension == 1:
    ph.show_curves(data_nda, labels=data_labels, fig_number=1,
                   linestyle=linestyles, markers=ph.MARKERS)

elif dimension == 2:
    ph.show_arrays(data_nda, title=data_labels, fig_number=1)

elif dimension == 3:
    sitkh.show_sitk_image(data_sitk, label=data_labels)

for m in range(0, len(monitors)):
    monitors[m].compute_measures()
    print("Computational time %s: %s" %
          (monitors[m].get_name(), monitors[m].get_computational_time()))

for k in measures_dic.keys():
    title = k
    x = []
    y = []
    legend = []
    for m in range(0, len(monitors)):
        y.append(monitors[m].get_measures()[title])
        legend.append(monitors[m].get_name())
        x.append(range(0, len(y[-1])))
    ph.show_curves(y, x=x, xlabel="iteration", labels=legend, title=title,
                   linestyle=linestyles, markers=ph.MARKERS, markevery=1)
