##
# \file admm_linear_solver_parameter_study.py
# \brief Class to run parameter study for ADMMLinearSolver
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import numpy as np

import pysitk.python_helper as ph
from nsol.solver_parameter_study import SolverParameterStudy
import nsol.admm_linear_solver as admm


class ADMMLinearSolverParameterStudy(SolverParameterStudy):

    ##
    # Store information for parameter study
    # \date       2017-09-13 23:16:59+0100
    #
    # \param      self                 The object
    # \param      solver               Solver object
    # \param      observer             Observer object
    # \param      dir_output           Output directory to write study results
    # \param      name                 Name of study, string (no white spaces)
    # \param      parameters           Dictionary holding information on
    #                                  varying parameter to sweep through
    #                                  during parameter study
    # \param      reconstruction_info  Dictionary holding information useful
    #                                  for later reconstruction, e.g. image
    #                                  space or sitk.Image to include image
    #                                  header information
    #
    def __init__(self,
                 solver,
                 observer,
                 dir_output,
                 name="ADMM",
                 parameters={
                     "alpha": np.arange(0.01, 0.05, 0.01),
                     "rho": np.arange(0.1, 1.5, 0.5),
                     # "data_loss": ["linear", "arctan"],
                     # "data_loss_scale": [1., 1.2],
                 },
                 reconstruction_info={},
                 append=False,
                 ):

        if not isinstance(solver, admm.ADMMLinearSolver):
            raise TypeError("solver must be of type 'ADMMLinearSolver'")

        super(self.__class__, self).__init__(
            solver=solver,
            parameters=parameters,
            observer=observer,
            dir_output=dir_output,
            name=name,
            reconstruction_info=reconstruction_info,
            append=append,
        )

    def _get_fileheader(self):

        # keys referring to the information to be printed in the file
        keys = ["alpha",
                "rho",
                "iterations",
                "minimizer",
                "iter_max",
                "x_scale",
                "data_loss",
                "data_loss_scale",
                "dimension",
                ]

        header = "## " + self._name
        for key in keys:
            # Only write information to header which stays constant in study
            if key not in self._parameters.keys():
                header += ", %s=%s" \
                    % (key, eval("str(self._solver.get_" + key + "())"))
        header += " (%s)" % (ph.get_time_stamp())
        header += "\n"
        return header
