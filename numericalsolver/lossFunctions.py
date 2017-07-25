##
# \file lossFunctions.py
# \brief      Collection of loss functions and mapping from residual to cost
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


##
# Gets the cost \f$C(\vec{x})\f$ from the residual \f$\vec{f}(\vec{x})\f$.
# \date       2017-07-14 11:42:18+0100
#
# \param      f      residual f, m-dimensional numpy array
# \param      loss   Choice for loss function rho
#
# \return     The cost from residual as scalar value >= 0
#
def get_ell2_cost_from_residual(f, loss="linear"):
    cost = 0.5 * np.sum(get_loss[loss](f**2))
    return cost


##
# Gets the gradient
# \f$
# \nabla C\f$ of the cost function given the residual
# \f$\vec{f}(\vec{x})\f$ and its Jacobian
# \f$\frac{d\vec{f}}{d\vec{x}}(\vec{x})\f$.
# \date       2017-07-14 11:30:11+0100
#
# \param      f      residual f, m-dimensional numpy array
# \param      jac_f  Jacobian of residual f, (m x n)-dimensional numpy array
# \param      loss   Choice for loss function rho
#
# \return     The gradient of the cost from residual as n-dimensional numpy
#             array
#
def get_gradient_ell2_cost_from_residual(f, jac_f, loss="linear"):
    grad = np.sum((get_gradient_loss[loss](f**2) * f)[:, np.newaxis] * jac_f,
                  axis=0)
    return grad


##
# Return linear loss, i.e. rho(f2) = f2
# \date       2017-07-25 19:34:23+0100
#
# \param      f2    scalar or numpy array; meant to be squared residual
#
# \return     rho(f2) as scalar or numpy array
#
def linear(f2):
    return f2


##
# Return gradient/derivative of linear loss, i.e. rho'(f2) = 1
# \date       2017-07-25 19:36:29+0100
#
# \param      f2    scalar or numpy array; meant to be squared residual
#
# \return     rho'(f2) as scalar or numpy array
#
def gradient_linear(f2):
    return np.ones_like(f2).astype(np.float64)


##
# Return soft ell1 approximation, i.e. rho(f2) = 2 (sqrt(1 + f2) - 1)
# \date       2017-07-25 19:37:52+0100
#
# \param      f2    scalar or numpy array; meant to be squared residual
#
# \return     rho(f2) as scalar or numpy array
#
def soft_l1(f2):
    return 2. * (np.sqrt(1.+f2) - 1.)


##
# Return gradient/derivative of soft ell1 loss, i.e. rho'(f2) = 1 / sqrt(1+f2)
# \date       2017-07-25 19:36:29+0100
#
# \param      f2    scalar or numpy array; meant to be squared residual
#
# \return     rho'(f2) as scalar or numpy array
#
def gradient_soft_l1(f2):
    return 1. / np.sqrt(1.+f2)


##
# Return huber loss
# \date       2017-07-25 19:37:52+0100
#
# \param      f2    scalar or numpy array; meant to be squared residual
#
# \return     rho(f2) as scalar or numpy array
#
def huber(f2, gamma=1.345):
    gamma = float(gamma)
    gamma2 = gamma * gamma
    return np.where(f2 < gamma2, f2, 2.*gamma*np.sqrt(f2)-gamma2)


##
# Return gradient/derivative of linear loss, i.e. rho'(f2)
# \date       2017-07-25 19:36:29+0100
#
# \param      f2    scalar or numpy array; meant to be squared residual
#
# \return     rho'(f2) as scalar or numpy array
#
def gradient_huber(f2, gamma=1.345):
    gamma = float(gamma)
    gamma2 = gamma * gamma
    return np.where(f2 < gamma2, 1., gamma/np.sqrt(f2))


##
# Return cauchy loss, i.e. rho(f2) = ln(1 + f2)
# \date       2017-07-25 19:37:52+0100
#
# \param      f2    scalar or numpy array; meant to be squared residual
#
# \return     rho(f2) as scalar or numpy array
#
def cauchy(f2):
    return np.log1p(f2)


##
# Return gradient/derivative of cauchy loss, i.e. rho'(f2) = 1 / (1 + f2)
# \date       2017-07-25 19:36:29+0100
#
# \param      f2    scalar or numpy array; meant to be squared residual
#
# \return     rho'(f2) as scalar or numpy array
#
def gradient_cauchy(f2):
    return 1. / (1. + f2)


##
# Return arctan loss, i.e. rho(f2) = arctan(f2)
# \date       2017-07-25 19:37:52+0100
#
# \param      f2    scalar or numpy array; meant to be squared residual
#
# \return     rho(f2) as scalar or numpy array
#
def arctan(f2):
    return np.arctan(f2)


##
# Return gradient/derivative of arctan loss, i.e. rho'(f2) = 1 / (1 + f2^2)
# \date       2017-07-25 19:36:29+0100
#
# \param      f2    scalar or numpy array; meant to be squared residual
#
# \return     rho'(f2) as scalar or numpy array
#
def gradient_arctan(f2):
    return 1. / (1. + f2**2)


# Dictionary for all loss functions
get_loss = {
    "linear": linear,
    "soft_l1": soft_l1,
    "huber": huber,
    "cauchy": cauchy,
    "arctan": arctan,
}

# Dictionary for all gradient loss functions
get_gradient_loss = {
    "linear": gradient_linear,
    "soft_l1": gradient_soft_l1,
    "huber": gradient_huber,
    "cauchy": gradient_cauchy,
    "arctan": gradient_arctan,
}
