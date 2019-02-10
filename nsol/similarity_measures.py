##
# \file similarity_measures.py
# \brief      Collection of similarity (and dissimilarity) functions
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#


import skimage.measure
import numpy as np


class SimilarityMeasures(object):

    ##
    # Compute sum of absolute differences (symmetric)
    # \date       2017-08-04 10:09:05+0100
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    #
    # \return     sum of absolute differences as scalar value >= 0
    #
    @staticmethod
    def sum_of_absolute_differences(x, x_ref):
        if x.shape != x_ref.shape:
            raise ValueError("Input data shapes do not match")
        return np.sum(np.abs(x - x_ref))

    ##
    # Compute mean of absolute error (symmetric)
    # \date       2019-02-10 11:29:07+0000
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    #
    # \return     mean of absolute error as scalar value >= 0
    #
    @staticmethod
    def mean_absolute_error(x, x_ref):
        mae = SimilarityMeasures.sum_of_absolute_differences(x, x_ref)
        mae /= float(x.size)
        return mae

    ##
    # Compute sum of squared differences (symmetric)
    # \date       2017-08-04 10:09:05+0100
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    #
    # \return     sum of squared differences as scalar value >= 0
    #
    @staticmethod
    def sum_of_squared_differences(x, x_ref):
        if x.shape != x_ref.shape:
            raise ValueError("Input data shapes do not match")
        return np.sum(np.square(x - x_ref))

    ##
    # Compute mean of squared error (symmetric)
    # \date       2017-08-04 10:09:46+0100
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    #
    # \return     mean of squared error as scalar value >= 0
    #
    @staticmethod
    def mean_squared_error(x, x_ref):
        mse = SimilarityMeasures.sum_of_squared_differences(x, x_ref)
        mse /= float(x.size)
        return mse

    ##
    # Compute root mean square error (symmetric)
    # \date       2017-08-04 10:09:46+0100
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    #
    # \return     mean of squared error as scalar value >= 0
    #
    @staticmethod
    def root_mean_square_error(x, x_ref):
        return np.sqrt(SimilarityMeasures.mean_squared_error(x, x_ref))

    ##
    # Compute peak signal to noise ratio (non-symmetric)
    # \date       2017-08-04 10:10:13+0100
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    #
    # \return     peak signal to noise ratio as scalar value
    #
    @staticmethod
    def peak_signal_to_noise_ratio(x, x_ref):
        mse = SimilarityMeasures.mean_squared_error(x, x_ref)
        return 10 * np.log10(np.max(x_ref) ** 2 / mse)

    ##
    # Compute normalized cross correlation (symmetric)
    # \date       2017-08-04 10:12:14+0100
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    #
    # \return     Normalized cross correlation as scalar value between -1 and 1
    #
    @staticmethod
    def normalized_cross_correlation(x, x_ref):
        if x.shape != x_ref.shape:
            raise ValueError("Input data shapes do not match")

        ncc = np.sum((x - x.mean()) * (x_ref - x_ref.mean()))
        ncc /= float(x.size * x.std(ddof=1) * x_ref.std(ddof=1))

        return ncc

    ##
    # Compute structural similarity (symmetric)
    # \see        Wang, Z. et al., 2004. Image Quality Assessment: From Error
    #             Visibility to Structural Similarity. IEEE Transactions on Image
    #             Processing, 13(4), pp.600-612.
    # \date       2017-08-04 10:16:32+0100
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    #
    # \return     Structural similarity as scalar value between 0 and 1
    #
    @staticmethod
    def structural_similarity(x, x_ref):
        return skimage.measure.compare_ssim(x, x_ref)

    ##
    # Compute Shannon entropy
    #
    # Shannon entropy H(X) = - sum p(x) * ln(p(x))
    # \see        Pluim, J.P.W., Maintz, J.B.A. & Viergever, M.A., 2003.
    #             Mutual-information-based registration of medical images: a
    #             survey. IEEE Transactions on Medical Imaging, 22(8), pp.986-1004.
    # \date       2017-08-04 10:21:02+0100
    #
    # \param      x     numpy data array
    # \param      bins  number of bins for histogram, int
    #
    # \return     Shannon entropy as scalar value \in [0, log_b(n)] (e.g.
    #             Wikipedia)
    #
    @staticmethod
    def shannon_entropy(x, bins=100):

        # histogram is computed over flattened array
        hist, bin_edges = np.histogram(x, bins=bins)

        # Compute probabilities
        prob = hist / float(np.sum(hist))

        entropy = - sum([p * np.log(p) for p in prob.flatten() if p != 0])

        return entropy

    ##
    # Compute joint entropy (symmetric)
    #
    # Joint entropy H(X,Y) = - sum p(x,y) * ln(p(x,y))
    # \see        Pluim, J.P.W., Maintz, J.B.A. & Viergever, M.A., 2003.
    #             Mutual-information-based registration of medical images: a
    #             survey. IEEE Transactions on Medical Imaging, 22(8), pp.986-1004.
    # \date       2017-08-04 10:35:18+0100
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    # \param      bins   number of bins for histogram, sequence or int
    #
    # \return     Joint entropy as scalar value >=0
    #
    @staticmethod
    def joint_entropy(x, x_ref, bins=100):

        hist, x_edges, y_edges = np.histogram2d(
            x.flatten(), x_ref.flatten(), bins=bins)

        # Compute probabilities
        prob = hist / float(np.sum(hist))

        jentropy = - sum([p * np.log(p) for p in prob.flatten() if p != 0])
        return jentropy

    ##
    # Compute mutual information (symmetric)
    #
    # MI(X,Y) = - sum p(x,y) * ln( p(x,y) / (p(x) * p(y)) ) = H(X) + H(Y) - H(X,Y)
    # \see        Pluim, J.P.W., Maintz, J.B.A. & Viergever, M.A., 2003.
    #             Mutual-information-based registration of medical images: a
    #             survey. IEEE Transactions on Medical Imaging, 22(8), pp.986-1004.
    # \see        Skouson, M.B., Quji Guo & Zhi-Pei Liang, 2001. A bound on mutual
    #             information for image registration. IEEE Transactions on Medical
    #             Imaging, 20(8), pp.843-846.
    # \date       2017-08-04 10:40:35+0100
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    # \param      bins   number of bins for histogram, sequence or int
    #
    # \return     Mutual information as scalar value >= 0 with upper bound as in
    #             Skouson2001
    #
    @staticmethod
    def mutual_information(x, x_ref, bins=100):
        mi = SimilarityMeasures.shannon_entropy(x, bins=bins)
        mi += SimilarityMeasures.shannon_entropy(x_ref, bins=bins)
        mi -= SimilarityMeasures.joint_entropy(x, x_ref, bins=bins)
        return mi

    ##
    # Compute mutual information (symmetric)
    #
    # NMI(X,Y) = H(X) + H(Y) / H(X,Y)
    # \see        Pluim, J.P.W., Maintz, J.B.A. & Viergever, M.A., 2003.
    #             Mutual-information-based registration of medical images: a
    #             survey. IEEE Transactions on Medical Imaging, 22(8), pp.986-1004.
    # \date       2017-08-04 10:40:35+0100
    #
    # \param      x      numpy data array
    # \param      x_ref  reference numpy data array
    # \param      bins   number of bins for histogram, sequence or int
    #
    # \return     Normalized mutual information as scalar value >= 0
    #
    @staticmethod
    def normalized_mutual_information(x, x_ref, bins=100):
        nmi = SimilarityMeasures.shannon_entropy(x, bins=bins)
        nmi += SimilarityMeasures.shannon_entropy(x_ref, bins=bins)
        nmi /= SimilarityMeasures.joint_entropy(x, x_ref, bins=bins)
        return nmi

    ##
    # Compute Dice's score (Dice's coefficient).
    #
    # dice(A, B) = 2 * |A \cap B | / (|A| + |B|)
    # \see        Dice, L.R., 1945. Measures of the Amount of Ecologic Association
    #             Between Species. Ecology, 26(3), pp.297-302.
    # \date       2017-08-04 11:11:21+0100
    #
    # \param      x      numpy data array, bool
    # \param      x_ref  reference numpy data array, bool
    #
    # \return Dice score between 0 and 1
    #
    @staticmethod
    def dice_score(x, x_ref):

        if x.dtype is not np.dtype(np.bool) or \
                x_ref.dtype is not np.dtype(np.bool):
            raise ValueError("x and x_ref need to be of type boolean")

        dice = 2 * np.sum(x * x_ref)
        dice /= np.sum(x) + np.sum(x_ref)

        return dice

    # Dictionary for all similarity measures
    similarity_measures = {
        "SSD": sum_of_squared_differences.__func__,
        "MAE": mean_absolute_error.__func__,
        "MSE": mean_squared_error.__func__,
        "RMSE": root_mean_square_error.__func__,
        "PSNR": peak_signal_to_noise_ratio.__func__,
        "SSIM": structural_similarity.__func__,
        "NCC": normalized_cross_correlation.__func__,
        "MI": mutual_information.__func__,
        "NMI": normalized_mutual_information.__func__,
    }

    # Values for each similarity measure to 'define' undefined states
    UNDEF = {
        "SSD": np.NaN,
        "MAE": np.NaN,
        "MSE": np.NaN,
        "RMSE": np.NaN,
        "PSNR": np.NaN,
        "SSIM": np.NaN,
        "NCC": np.NaN,
        "MI": np.NaN,
        "NMI": np.NaN,
    }
