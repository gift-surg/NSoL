##
# \file input_arparser.py
# \brief      Class holding a collection of possible arguments to parse for
#             scripts
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2017
#

import six
import inspect
import argparse

import pysitk.python_helper as ph

from nsol.similarity_measures import SimilarityMeasures as \
    SimilarityMeasures
from nsol.loss_functions import LossFunctions as \
    LossFunctions
from nsol.definitions import ALLOWED_INPUT_FILE_EXTENSIONS
from nsol.definitions import ALLOWED_NOISE_TYPES

# Allowed input file types
INPUT_FILE_TYPES = "(" + (", ").join(ALLOWED_INPUT_FILE_EXTENSIONS) + ")"
NOISE_TYPES = "(" + (", ").join(ALLOWED_NOISE_TYPES) + ", or none)"

##
# Class holding a collection of possible arguments to parse for scripts
# \date       2017-08-07 01:26:11+0100
#


class InputArgparser(object):

    def __init__(self,
                 description=None,
                 prog=None,
                 epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
                 ):

        kwargs = {}
        if description is not None:
            kwargs['description'] = description
        if prog is not None:
            kwargs['prog'] = prog
        if epilog is not None:
            kwargs['epilog'] = epilog

        self._parser = argparse.ArgumentParser(**kwargs)

    def get_parser(self):
        return self._parser

    def parse_args(self):
        return self._parser.parse_args()

    def print_arguments(self, args, title="Input Parameters:"):
        ph.print_title(title)
        for arg in sorted(vars(args)):
            ph.print_info("%s: " % (arg), newline=False)
            print(getattr(args, arg))

    def add_observation(
        self,
        option_string="--observation",
        type=str,
        help="Path to observation %s." % (INPUT_FILE_TYPES),
        required=True,
    ):
        self._add_argument(dict(locals()))

    def add_filename(
        self,
        option_string="--filename",
        type=str,
        help="Path to filename %s." % (INPUT_FILE_TYPES),
        required=True,
    ):
        self._add_argument(dict(locals()))

    def add_dir_input(
        self,
        option_string="--dir-input",
        type=str,
        help="Input directory.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_result(
        self,
        option_string="--result",
        type=str,
        help="Specify path for obtained result %s." % (INPUT_FILE_TYPES),
        required=True,
    ):
        self._add_argument(dict(locals()))

    def add_reconstruction_type(
        self,
        option_string="--reconstruction-type",
        type=str,
        help="Define reconstruction type. Allowed values are "
        "'TVL1', 'TVL2', 'HuberL1' and 'HuberL2'.",
        default="TVL1",
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_measures(
        self,
        option_string="--measures",
        type=str,
        nargs="+",
        help="Measures to be evaluated between reference (if given) and "
        "reconstruction %s. " % ("(" + (", ").join(
            SimilarityMeasures.similarity_measures.keys()) + ")"),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_reference(
        self,
        option_string="--reference",
        type=str,
        help="Path to reference %s. Similarity measures are "
        "computed only if reference is given." % (INPUT_FILE_TYPES),
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_alpha_range(
        self,
        option_string="--alpha-range",
        nargs="+",
        type=float,
        help="Specify regularization parameter array by providing "
        "'First Last Step' information. Array is generated according to "
        "np.linspace(First, Last, Step).",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_data_losses(
        self,
        option_string="--data-losses",
        nargs="+",
        help="Specify data losses to be used %s. " % ("(" + (", ").join(
            LossFunctions.get_loss.keys()) + ")"),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_data_loss_scale_range(
        self,
        option_string="--data-loss-scale-range",
        nargs="+",
        type=float,
        help="Specify data loss scales by providing 'First Last Step' "
        "information. Array is generated according to "
        "np.linspace(First, Last, Step).",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_study_name(
        self,
        option_string="--study-name",
        type=str,
        help="Name of parameter study without white spaces.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_dir_output(
        self,
        option_string="--dir-output",
        type=str,
        help="Output directory.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_dir_output_figures(
        self,
        option_string="--dir-output-figures",
        type=str,
        help="If given, created figures are saved to this directory.",
        default=None,
    ):
        self._add_argument(dict(locals()))

    def add_alpha(
        self,
        option_string="--alpha",
        nargs="+",
        type=float,
        help="Regularization parameter alpha to solve minimization problem "
        "min_x [f(x) + alpha g(x)] with f denoting the data and g the "
        "regularization term, respectively.",
        default=0.03,
    ):
        self._add_argument(dict(locals()))

    def add_blur(
        self,
        option_string="--blur",
        type=float,
        nargs="+",
        help="Specify for Gaussian blurring a single standard deviation "
        "(isotropic blurring) or the standard deviation in each spatial "
        "direction (elliptic blurring)",
        default=0,
    ):
        self._add_argument(dict(locals()))

    def add_noise(
        self,
        option_string="--noise",
        type=str,
        help="Specify type of noise to be applied %s." % NOISE_TYPES,
        default=None,
    ):
        self._add_argument(dict(locals()))

    def add_noise_level(
        self,
        option_string="--noise-level",
        type=float,
        help="Specify noise level to be applied.",
        default=None,
    ):
        self._add_argument(dict(locals()))

    def add_colormap(
        self,
        option_string="--colormap",
        type=str,
        help="Specify colormap type to be applied for visualization "
        "(Only for 2D). "
        "E.g. 'Greys_r', 'jet'",
        default=None,
    ):
        self._add_argument(dict(locals()))

    def add_iter_max(
        self,
        option_string="--iter-max",
        type=int,
        help="Number of maximum iterations for the numerical solver.",
        default=10,
    ):
        self._add_argument(dict(locals()))

    def add_minimizer(
        self,
        option_string="--minimizer",
        type=str,
        help="Choice of minimizer used for the inverse problem associated to "
        "the f(x) + alpha g(x). Possible choices are 'lsmr' or any solver in "
        "scipy.optimize.minimize like 'L-BFGS-B'. Note, in case of a chosen "
        "non-linear data loss only non-linear solvers like 'L-BFGS-B' are "
        "viable.",
        default="lsmr",
    ):
        self._add_argument(dict(locals()))

    def add_rho(
        self,
        option_string="--rho",
        type=float,
        help="Regularization parameter for augmented Lagrangian term for "
        "TV regularization by ADMM.",
        default=0.5,
    ):
        self._add_argument(dict(locals()))

    def add_iterations(
        self,
        option_string="--iterations",
        type=int,
        help="Number of ADMM/Primal Dual iterations.",
        default=10,
    ):
        self._add_argument(dict(locals()))

    def add_solver(
        self,
        option_string="--solver",
        type=str,
        help="Type of solver. Either 'ADMM' or 'PD' to choose between "
        "Alternating Direction Method of Multipliers (ADMM) or "
        "Primal-Dual (PD) approach, respectively.",
        default="PD",
    ):
        self._add_argument(dict(locals()))

    def add_data_loss(
        self,
        option_string="--data-loss",
        type=str,
        help="Loss function rho used for data term, i.e. rho((y_k - A_k x)^2) "
        "Possible choices are 'linear', 'soft_l1, 'huber', 'arctan' and "
        "'cauchy'.",
        default="linear",
    ):
        self._add_argument(dict(locals()))

    def add_data_loss_scale(
        self,
        option_string="--data-loss-scale",
        type=float,
        help="Value of soft margin between inlier and outlier residuals, "
        "default is 1.0. The loss function is evaluated as "
        "rho_(f2) = C**2 * rho(f2 / C**2), where C is data_loss_scale. "
        "This parameter has no effect with data_loss='linear', but for other "
        "loss values it is of crucial importance.",
        default=1,
    ):
        self._add_argument(dict(locals()))

    def add_pd_alg_type(
            self,
            option_string="-pd_alg_type",
            type=str,
            help="Algorithm used to dynamically update parameters for each "
            "iteration of the dual algorithm. "
            "Possible choices are 'ALG2', 'ALG2_AHMOD' and 'ALG3' as "
            "described in Chambolle et al., 2011",
            default="ALG2",
    ):
        self._add_argument(dict(locals()))

    def add_verbose(
        self,
        option_string="--verbose",
        type=int,
        help="Turn on/off verbose output.",
        default=1,
    ):
        self._add_argument(dict(locals()))

    def add_option(
        self,
        option_string="--option",
        nargs=None,
        type=float,
        help="Add option.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    ##
    # Adds an argument to argument parser.
    #
    # Rationale: Make interface as generic as possible so that function call
    # works regardless the name of the desired option
    # \date       2017-08-06 21:54:51+0100
    #
    # \param      self     The object
    # \param      allvars  all variables set at respective function call as
    #                      dictionary
    #
    def _add_argument(self, allvars):

        # Skip variable 'self'
        allvars.pop('self')

        # Get name of argument to add
        option_string = allvars.pop('option_string')

        # Build dictionary for additional, optional parameters
        kwargs = {}
        for key, value in six.iteritems(allvars):
            kwargs[key] = value

        # Add information on default value in case provided
        if 'default' in kwargs.keys():

            txt_default = " [default: %s]" % (str(kwargs['default']))

            # Case where 'required' key is given:
            if 'required' in kwargs.keys():

                # Only add information in case argument is not mandatory to
                # parse
                if kwargs['default'] is not None and not kwargs['required']:
                    kwargs['help'] += txt_default

            # Case where no such field was provided
            else:
                if kwargs['default'] is not None:
                    kwargs['help'] += txt_default

        # Add argument with its options
        self._parser.add_argument(option_string, **kwargs)
