#!/usr/bin/python

##
# \file showParameterStudy.py
# \brief      Script to visualize performed parameter studies.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2017
#

# Import libraries
import SimpleITK as sitk
import argparse
import numpy as np
import sys
import os
from itertools import repeat

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
import nsol.ReaderParameterStudy as ReaderParameterStudy
import nsol.InputArgparser as InputArgparser


def show_L_curve(parameter_study_reader, lines, dir_output=None):
    name = parameter_study_reader.get_parameter_study_name()

    line_to_parameter_labels_dic = \
        parameter_study_reader.get_line_to_parameter_labels()
    nda_data = parameter_study_reader.get_results("Data")
    nda_reg = parameter_study_reader.get_results("Reg")

    labels = []
    y = []
    x = []
    markers = ph.MARKERS*100

    linestyle = []
    linestyle_curve = []
    y_curve = []
    x_curve = []
    labels_curve = []
    markers_curve = []

    for j, line in enumerate(lines):
        # Get labels
        labels.extend([line_to_parameter_labels_dic[i] for i in line])

        # Get arrays to plot
        x.extend([nda_data[i, -1] for i in line])
        y.extend([nda_reg[i, -1] for i in line])

        # duplicate linestyle element len(line)-amount of times in list
        linestyle.extend(
            [i for item in ph.LINESTYLES[0:-1]
             for i in repeat(item, len(line))])

        # Only if connecting curve is desired
        x_curve.append([nda_data[i, -1] for i in line])
        y_curve.append([nda_reg[i, -1] for i in line])
        labels_curve.append(None)
        linestyle_curve.append((10*ph.LINESTYLES[0:-1])[j])
        markers_curve.append("None")

    # Build connecting curve
    x_curve.extend([i] for i in x)
    y_curve.extend([i] for i in y)
    labels_curve.extend(i for i in labels)
    linestyle_curve.extend(i for i in linestyle)
    markers_curve.extend(i for i in markers)

    # Plot
    xlabel = "Data"
    ylabel = "Regularizer"
    title = "%s: L-curve" % name
    ph.show_curves(
        # y=y, x=x, linestyle=linestyle,  # no connecting curve
        # markers=markers, labels=labels,
        y=y_curve, x=x_curve, linestyle=linestyle_curve,  # connecting curve
        markers=markers_curve, labels=labels_curve,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        markevery=1,
        y_axis_style="loglog",
        # y_axis_style="semilogy",
        filename=name+"_L-curve.pdf",
        directory=dir_output,
        save_figure=0 if dir_output is None else 1,
    )


def show_measures(parameter_study_reader, lines, dir_output=None):
    name = parameter_study_reader.get_parameter_study_name()

    line_to_parameter_labels_dic = \
        parameter_study_reader.get_line_to_parameter_labels()

    # Plot
    for m in parameter_study_reader.get_measures():
        y = []
        linestyle = []
        labels = []
        for line in lines:
            nda = parameter_study_reader.get_results(m)

            # Store all iterations for current parameter
            y.extend([nda[i, :] for i in line])
            labels.extend([line_to_parameter_labels_dic[i] for i in line])

            # duplicate linestyle element len(line)-amount of times in list
            linestyle.extend(
                [i for item in ph.LINESTYLES[0:-1]
                 for i in repeat(item, len(line))])

        xlabel = "iteration"
        # labels = m
        title = "%s: %s" % (name, m)
        ph.show_curves(y,
                       xlabel=xlabel,
                       # ylabel=ylabel,
                       linestyle=linestyle,
                       labels=labels,
                       title=title,
                       markers=ph.MARKERS*100,
                       markevery=10,
                       filename=name+"_"+m+".pdf",
                       directory=dir_output,
                       save_figure=0 if dir_output is None else 1,
                       )


def show_reconstructions(parameter_study_reader,
                         lines,
                         dir_output=None,
                         colormap="Greys_r",
                         ):

    try:
        reconstructions_dic = parameter_study_reader.get_reconstructions()
    except IOError as e:
        print("Error: '%s'. Visualization skipped." % e)
        return

    name = parameter_study_reader.get_parameter_study_name()
    line_to_parameter_labels_dic = \
        parameter_study_reader.get_line_to_parameter_labels()

    for j, line in enumerate(lines):
        # Get labels
        labels = [line_to_parameter_labels_dic[i] for i in line]

        data_nda = [reconstructions_dic[str(ell)].reshape(
            reconstructions_dic["shape"]) for ell in line]

        if len(reconstructions_dic["shape"]) == 2:
            try:
                if len(lines) == 1:
                    filename = name + "_reconstructions.pdf"
                else:
                    filename = name + "_reconstructions_%d.pdf" % (j+1)

                ph.show_arrays(data_nda,
                               title=labels,
                               fig_number=None,
                               cmap=colormap,
                               use_same_scaling=True,
                               # fontsize=8,
                               directory=dir_output,
                               filename=filename,
                               save_figure=0 if dir_output is None else 1,
                               )
            except ValueError as e:
                print("Error '%s'. Visualization skipped." % e)
                return

        elif len(reconstructions_dic["shape"]) == 3:
            origin = reconstructions_dic["origin"]
            spacing = reconstructions_dic["spacing"]
            direction = reconstructions_dic["direction"]
            recons_sitk = []

            for nda in data_nda:
                recon_sitk = sitk.GetImageFromArray(nda)
                recon_sitk.SetSpacing(spacing)
                recon_sitk.SetOrigin(origin)
                recon_sitk.SetDirection(direction)
                recons_sitk.append(recon_sitk)
            labels = [line.replace(".", "p") for line in labels]
            sitkh.show_sitk_image(recons_sitk, label=labels)

if __name__ == '__main__':

    # Read input
    input_parser = InputArgparser.InputArgparser(
        description="Show and analyse stored parameter study.",
        prog="python " + os.path.basename(__file__),
    )

    input_parser.add_dir_input(
        help="Input directory where parameter study results are located.",
        required=True)
    input_parser.add_study_name(required=True)
    input_parser.add_dir_output_figures()
    input_parser.add_colormap(default="Greys_r")

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    parameter_study_reader = ReaderParameterStudy.ReaderParameterStudy(
        directory=args.dir_input, name=args.study_name)
    parameter_study_reader.read_study()

    parameters_dic = parameter_study_reader.get_parameters()
    parameters_to_line_dic = parameter_study_reader.get_parameters_to_line()
    line_to_parameter_labels_dic = \
        parameter_study_reader.get_line_to_parameter_labels()

    # ---------------------- Get lines to varying alpha ----------------------
    # Get dictionary with single varying key
    lines = []
    if len(parameters_dic.keys()) == 1:
        p = parameters_dic

        # Get lines in result files associated to varying 'alpha'
        lines.append(parameter_study_reader.get_lines_to_parameters(p))

    else:
        for k in parameters_dic.keys():
            if k == "alpha":
                continue
            for i, val in enumerate(parameters_dic[k]):
                p = {"alpha": parameters_dic["alpha"]}
                p[k] = parameters_dic[k][i]

                # Get lines in result files associated to varying 'alpha'
                lines.append(parameter_study_reader.get_lines_to_parameters(p))

    show_L_curve(parameter_study_reader, lines, args.dir_output_figures)
    show_measures(parameter_study_reader, lines, args.dir_output_figures)
    show_reconstructions(parameter_study_reader, lines,
                         args.dir_output_figures, colormap=args.colormap)
