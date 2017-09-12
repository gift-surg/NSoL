##
# \file PrimalDualSolverParameterStudy.py
# \brief Class to run parameter study for PrimalDualSolver
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import numpy as np

import pythonhelper.PythonHelper as ph
from numericalsolver.SolverParameterStudy import SolverParameterStudy
import numericalsolver.PrimalDualSolver as pd


class PrimalDualSolverParameterStudy(SolverParameterStudy):

    def __init__(self,
                 solver,
                 observer,
                 dir_output,
                 name="PrimalDual",
                 parameters={
                     "alpha": np.arange(0.01, 0.05, 0.005),
                     "alg_type": ["ALG2", "ALG2_AHMOD", "ALG3"],
                 },
                 reconstruction_info_dic={},
                 ):

        if not isinstance(solver, pd.PrimalDualSolver):
            raise TypeError("solver must be of type 'PrimalDualSolver'")

        super(self.__class__, self).__init__(
            solver=solver,
            parameters=parameters,
            observer=observer,
            dir_output=dir_output,
            name=name,
            reconstruction_info_dic=reconstruction_info_dic,
        )

    def _get_fileheader(self):

        # keys referring to the information to be printed in the file
        keys = ["alpha",
                "iterations",
                "x_scale",
                "L2"
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
