##
# \file TikhonovLinearSolverParameterStudy.py
# \brief
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import numpy as np

import pythonhelper.PythonHelper as ph
from numericalsolver.SolverParameterStudy import SolverParameterStudy
import numericalsolver.TikhonovLinearSolver as tk


class TikhonovLinearSolverParameterStudy(SolverParameterStudy):

    def __init__(self,
                 solver,
                 monitor,
                 dir_output,
                 name="Tikhonov",
                 parameters={
                     "alpha": np.arange(0.02, 0.5, 0.05),
                     # "data_loss": ["linear", "arctan"],
                     # "data_loss_scale": [1., 1.2],
                 },
                 ):

        if not isinstance(solver, tk.TikhonovLinearSolver):
            raise TypeError("solver must be of type 'TikhonovLinearSolver'")

        super(self.__class__, self).__init__(
            solver=solver, parameters=parameters, monitor=monitor,
            dir_output=dir_output, name=name)

    def _get_fileheader(self):

        header = "## " + self._name
        header += ", iter_max = %d" % (self._solver.get_iter_max())
        header += " (%s)" % (ph.get_time_stamp())
        header += "\n"
        return header
