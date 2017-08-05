##
# \file ADMMLinearSolverParameterStudy.py
# \brief
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
                 monitor,
                 dir_output,
                 name="ADMM",
                 parameters={
                     "alpha": np.arange(0.1, 0.5, 0.1),
                     "rho": np.arange(0.1, 1.5, 0.5),
                     # "data_loss": ["linear", "arctan"],
                     # "data_loss_scale": [1., 1.2],
                 },
                 ):

        if not isinstance(solver, admm.ADMMLinearSolver):
            raise TypeError("solver must be of type 'ADMMLinearSolver'")

        super(self.__class__, self).__init__(
            solver=solver, parameters=parameters, monitor=monitor,
            dir_output=dir_output, name=name)

    def _get_fileheader(self):

        header = "## " + self._name
        # header += ", iter_max = %d" % (self._solver.get_iter_max())
        header += " (%s)" % (ph.get_time_stamp())
        header += "\n"
        return header