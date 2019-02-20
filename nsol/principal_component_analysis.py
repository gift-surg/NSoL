# \file principal_component_analysis.py
# \brief      Principal Component Analysis (PCA) classes for dimensionality
#             reduction.
#
# Standard and robust PCA methods are available.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Feb 2019


import sys
import time
from numpy import *
from numpy.linalg import svd, norm
from multiprocessing.pool import ThreadPool

import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


##
# Class to compute principal components
# \date       2019-02-19 17:04:21+0000
#
class PrincipalComponentAnalysis(object):

    ##
    # Store data points for computation
    # \date       2019-02-19 17:04:40+0000
    #
    # \param      self    The object
    # \param      points  Numpy data array of shape n_points x dim, with dim
    #                     either 2 or 3
    #
    def __init__(self, points):

        points = np.array(points)

        if points.shape[1] not in [2, 3]:
            raise IOError(
                "Numpy array must be of shape N x dim, "
                "with dim either 2 or 3.")

        self._points = points

    ##
    # Perform PCA and store associated outcomes
    # \date       2019-02-19 17:05:28+0000
    #
    # \param      self  The object
    #
    def run(self):
        self._mean = np.mean(self._points, axis=0)
        self._cov = np.cov(self._points - self._mean, rowvar=False)

        # use 'eigh' rather than 'eig' since cov is symmetric;
        # performance gain can be substantial
        eigval, eigvec = np.linalg.eigh(self._cov)

        # eigenvalues in descending order
        idx = eigval.argsort()[::-1]
        self._eigval = eigval[idx]
        self._eigvec = eigvec[:, idx]

        # establish right-handed coordinate system
        self._eigvec[:, 2] = np.cross(self._eigvec[:, 0], self._eigvec[:, 1])

    def get_mean(self):
        return self._mean

    def get_cov(self):
        return self._cov

    def get_eigvec(self):
        return self._eigvec

    def get_eigval(self):
        return self._eigval

    def show(self, title, ax=None, step=1):
        points = self._points[::step, :]

        if ax is None:
            fig = plt.figure(title)
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            color="red", marker="x"
        )
        ax.set_title(title)

        # moment = scipy.stats.moment(self._points, moment=1)
        # ax.scatter(moment[0], moment[1], moment[2],
        #            color="green", marker="o")

        colors = ["g", "b", "k"]
        for i in range(0, len(self._mean)):
            ax.quiver(
                self._mean[0], self._mean[1], self._mean[2],
                self._eigval[i] * self._eigvec[0, i],
                self._eigval[i] * self._eigvec[1, i],
                self._eigval[i] * self._eigvec[2, i],
                color=colors[i],
                label="eigvec%d" % (i + 1),
            )
        plt.legend()


##
# Robust PCA, based on Augmented Lagrange Multiplier (ALM) algorithm, see
# [Algorithm 1, Candes2011].
#
# Used for test purposes only. If used eventually, obey to LICENSE as mentioned
# at the github repo.
#
# \see        https://github.com/dganguli/robust-pca
# \see        https://stackoverflow.com/questions/40721260/how-to-use-robust-pca-output-as-principal-component-eigenvectors-from-traditio
# \date       2019-02-20 11:49:49+0000
#
class AlmRobustPrincipalComponentAnalysis(object):

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        # Matrix M = L + S

        # L low-rank; to be used for further eigenanalysis (standard PCA)
        self.L = Lk

        # S sparse (noise)
        self.S = Sk

        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')


##
# Robust PCA, based on ADMM algorithm.
#
# Used for test purposes only. If used eventually, obey to LICENSE as mentioned
# at the github repo.
#
# \see        https://github.com/jkarnows/rpcaADMM
# \date       2019-02-20 14:51:20+0000
#
class AdmmRobustPrincipalComponentAnalysis(object):

    def __init__(self, D):
        self._data = D

    @staticmethod
    def _prox_l1(v, lambdat):
        """
        The proximal operator of the l1 norm.

        prox_l1(v,lambdat) is the proximal operator of the l1 norm
        with parameter lambdat.

        Adapted from: https://github.com/cvxgrp/proximal/blob/master/matlab/prox_l1.m
        """

        return maximum(0, v - lambdat) - maximum(0, -v - lambdat)

    @staticmethod
    def _prox_matrix(v, lambdat, prox_f):
        """
        The proximal operator of a matrix function.

        Suppose F is a orthogonally invariant matrix function such that
        F(X) = f(s(X)), where s is the singular value map and f is some
        absolutely symmetric function. Then

        X = prox_matrix(V,lambdat,prox_f)

        evaluates the proximal operator of F via the proximal operator
        of f. Here, it must be possible to evaluate prox_f as prox_f(v,lambdat).

        For example,

        prox_matrix(V,lambdat,prox_l1)

        evaluates the proximal operator of the nuclear norm at V
        (i.e., the singular value thresholding operator).

        Adapted from: https://github.com/cvxgrp/proximal/blob/master/matlab/prox_matrix.m
        """

        U, S, V = svd(v, full_matrices=False)
        S = S.reshape((len(S), 1))
        pf = diagflat(prox_f(S, lambdat))
        # It should be V.conj().T given MATLAB-Python conversion, but matrix
        # matches with out the .T so kept it.
        return U.dot(pf).dot(V.conj())

    @staticmethod
    def _avg(*args):
        N = len(args)
        x = 0
        for k in range(N):
            x = x + args[k]
        x = x / N
        return x

    @staticmethod
    def _objective(X_1, g_2, X_2, g_3, X_3):
        """
        Objective function for Robust PCA:
            Noise - squared frobenius norm (makes X_i small)
            Background - nuclear norm (makes X_i low rank)
            Foreground - entrywise L1 norm (makes X_i small)
        """
        tmp = svd(X_3, compute_uv=0)
        tmp = tmp.reshape((len(tmp), 1))
        return norm(X_1, 'fro')**2 + g_2 * norm(hstack(X_2), 1) + g_3 * norm(tmp, 1)

    def run(self):
        """
        ADMM implementation of matrix decomposition. In this case, RPCA.

        Adapted from: http://web.stanford.edu/~boyd/papers/prox_algs/matrix_decomp.html
        """

        data = self._data

        # Create thread pool for asynchronous processing
        pool = ThreadPool(processes=3)

        N = 3         # the number of matrices to split into
        # (and cost function expresses how you want them)

        A = float_(data)    # A = S + L + V
        m, n = A.shape

        g2_max = norm(hstack(A).T, inf)
        g3_max = norm(A, 2)
        g2 = 0.15 * g2_max
        g3 = 0.15 * g3_max

        MAX_ITER = 100
        ABSTOL = 1e-4
        RELTOL = 1e-2

        start = time.time()

        lambdap = 1.0
        rho = 1.0 / lambdap

        X_1 = zeros((m, n))
        X_2 = zeros((m, n))
        X_3 = zeros((m, n))
        z = zeros((m, N * n))
        U = zeros((m, n))

        print('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' % (
            'iter',
            'r norm',
            'eps pri',
            's norm',
            'eps dual',
            'objective',
        ))

        # Saving state
        h = {}
        h['objval'] = zeros(MAX_ITER)
        h['r_norm'] = zeros(MAX_ITER)
        h['s_norm'] = zeros(MAX_ITER)
        h['eps_pri'] = zeros(MAX_ITER)
        h['eps_dual'] = zeros(MAX_ITER)

        def x1update(x, b, l):
            return (1.0 / (1.0 + l)) * (x - b)

        def x2update(x, b, l, g, pl):
            return pl(x - b, l * g)

        def x3update(x, b, l, g, pl, pm):
            return pm(x - b, l * g, pl)

        def update(func, item):
            return map(func, [item])[0]

        for k in range(MAX_ITER):

            B = self._avg(X_1, X_2, X_3) - A / N + U

            # Original MATLAB x-update
            # X_1 = (1.0/(1.0+lambdap))*(X_1 - B)
            # X_2 = prox_l1(X_2 - B, lambdap*g2)
            # X_3 = prox_matrix(X_3 - B, lambdap*g3, prox_l1)

            # Parallel x-update
            async_X1 = pool.apply_async(
                update, (lambda x: x1update(x, B, lambdap), X_1))
            async_X2 = pool.apply_async(
                update, (lambda x: x2update(x, B, lambdap, g2, self._prox_l1), X_2))
            async_X3 = pool.apply_async(update, (lambda x: x3update(
                x, B, lambdap, g3, self._prox_l1, self._prox_matrix), X_3))

            X_1 = async_X1.get()
            X_2 = async_X2.get()
            X_3 = async_X3.get()

            # (for termination checks only)
            x = hstack([X_1, X_2, X_3])
            zold = z
            z = x + tile(-self._avg(X_1, X_2, X_3) + A * 1.0 / N, (1, N))

            # u-update
            U = B

            # diagnostics, reporting, termination checks
            h['objval'][k] = self._objective(X_1, g2, X_2, g3, X_3)
            h['r_norm'][k] = norm(x - z, 'fro')
            h['s_norm'][k] = norm(-rho * (z - zold), 'fro')
            h['eps_pri'][k] = sqrt(m * n * N) * ABSTOL + \
                RELTOL * maximum(norm(x, 'fro'), norm(-z, 'fro'))
            h['eps_dual'][k] = sqrt(m * n * N) * ABSTOL + \
                RELTOL * sqrt(N) * norm(rho * U, 'fro')

            if (k == 0) or (mod(k + 1, 10) == 0):
                print('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' % (
                    k + 1,
                    h['r_norm'][k],
                    h['eps_pri'][k],
                    h['s_norm'][k],
                    h['eps_dual'][k],
                    h['objval'][k],
                ))
            if (h['r_norm'][k] < h['eps_pri'][k]) \
                    and (h['s_norm'][k] < h['eps_dual'][k]):
                break

        h['addm_toc'] = time.time() - start
        h['admm_iter'] = k

        # Matrix M = L + S + E
        # X1 ~ S, sparse
        h['X1_admm'] = X_1

        # X2 ~ E, error/noise
        h['X2_admm'] = X_2

        # X3 ~ L, low-rank (to be used for further eigenanalysis, standard PCA)
        h['X3_admm'] = X_3

        return h
