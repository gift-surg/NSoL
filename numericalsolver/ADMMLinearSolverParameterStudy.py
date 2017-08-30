##
# \file ADMMLinearSolverParameterStudy.py
# \brief Class to run parameter study for ADMMLinearSolver
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import numpy as np

import pythonhelper.PythonHelper as ph
from numericalsolver.SolverParameterStudy import SolverParameterStudy
import numericalsolver.ADMMLinearSolver as admm


class ADMMLinearSolverParameterStudy(SolverParameterStudy):

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
                 ):

        if not isinstance(solver, admm.ADMMLinearSolver):
            raise TypeError("solver must be of type 'ADMMLinearSolver'")

        super(self.__class__, self).__init__(
            solver=solver, parameters=parameters, observer=observer,
            dir_output=dir_output, name=name)

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
