##
# \file ParameterStudy.py
# \brief
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#


import pythonhelper.PythonHelper as ph
from numericalsolver.ParameterStudy import ParameterStudy
import numericalsolver.TikhonovLinearSolver as tk


class TikhonovParameterStudy(ParameterStudy):

    def __init__(self,
                 solver,
                 monitor,
                 dir_output,
                 name="Tikhonov",
                 alphas=[None],
                 data_losses=[None],
                 data_loss_scales=[None]):

        if not isinstance(solver, tk.TikhonovLinearSolver):
            raise TypeError("solver must be of type 'TikhonovLinearSolver'")

        parameters = {
            "alpha": alphas,
            "data_loss": data_losses,
            "data_loss_scale": data_loss_scales,
        }

        super(self.__class__, self).__init__(
            solver=solver, parameters=parameters, monitor=monitor,
            dir_output=dir_output, name=name)

    def _get_fileheader(self):

        header = "## " + self._name
        header += ", iter_max = %d" % (self._solver.get_iter_max())
        header += " (%s)" % (ph.get_time_stamp())
        header += "\n"
        return header
