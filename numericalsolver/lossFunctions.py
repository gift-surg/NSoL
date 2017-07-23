##
# \file lossFunctions.py
# \brief      Collection of loss functions
#
# The overall cost function is defined as
# \f[ C:\,\mathbb{R}^n \rightarrow \mathbb{R}_{\ge 0},\,\vec{x} \mapsto
# C(\vec{x}) := \frac{1}{2} \sum_{i=0}^{m-1} \rho( f_i(\vec{x})^2 )
# \f] with the loss function
# \f$\rho:\,\mathbb{R}\rightarrow\mathbb{R}_{\ge0}\f$ and the residual
# \f$ \vec{f}:\, \mathbb{R}^n \rightarrow
# \mathbb{R}^m,\,\vec{x}\mapsto\vec{f}(\vec{x}) = \b
# ig(f_0(\vec{x}),\,f_1(\vec{x}),\dots, f_{m-1}(\vec{x})\big)
# \f$
#
# \see        https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.least_squares.html
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#


import numpy as np


def linear(f):
    return f


def gradient_linear(f):
    return np.ones_like(f).astype(np.float64)


def soft_l1(f):
    return 2. * (np.sqrt(1.+f) - 1.)


def gradient_soft_l1(f):
    return 1. / np.sqrt(1.+f)


def huber(f, gamma=1.345):
    gamma = float(gamma)
    gamma2 = gamma * gamma
    return np.where(f < gamma2, f, 2.*gamma*np.sqrt(f)-gamma2)


def gradient_huber(f, gamma=1.345):
    gamma = float(gamma)
    gamma2 = gamma * gamma
    return np.where(f < gamma2, 1., gamma/np.sqrt(f))


def cauchy(f):
    return np.log1p(f)


def gradient_cauchy(f):
    return 1. / (1. + f)


def arctan(f):
    return np.arctan(f)


def gradient_arctan(f):
    return 1. / (1. + f**2)


get_loss = {
    "linear": linear,
    "soft_l1": soft_l1,
    "huber": huber,
    "cauchy": cauchy,
    "arctan": arctan,
}
get_gradient_loss = {
    "linear": gradient_linear,
    "soft_l1": gradient_soft_l1,
    "huber": gradient_huber,
    "cauchy": gradient_cauchy,
    "arctan": gradient_arctan,
}
