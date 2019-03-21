#!/usr/bin/python

##
# \file show_parameter_study.py
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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import repeat

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import nsol.data_reader as dr
import nsol.reader_parameter_study as ReaderParameterStudy
import nsol.input_argparser as InputArgparser


def show_L_curve(parameter_study_reader, lines, ctr, dir_output=None):
    name = parameter_study_reader.get_parameter_study_name()

    line_to_parameter_labels_dic = \
        parameter_study_reader.get_line_to_parameter_labels()
    nda_data = parameter_study_reader.get_results("Data")
    nda_reg = parameter_study_reader.get_results("Reg")

    labels = []
    y = []
    x = []
    markers = ph.MARKERS * 100

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

    # Plot
    fig = plt.figure(ph.add_one(ctr))
    fig.clf()
    ax = fig.gca()

    # Draw connecting line and an arrow to indicate increasing alpha
    l = plt.plot(x, y, color="lightgrey")
    start_ind = 0
    end_ind = 1
    l[0].axes.annotate(
        '',
        xytext=(x[start_ind], y[start_ind]),
        xy=(x[end_ind], y[end_ind]),
        arrowprops=dict(arrowstyle="->", color="lightgrey"),
        size=20,
    )

    for c in range(len(y)):
        l = plt.plot(x[c], y[c], label=labels[c])

        # Extract line object to adjust line settings
        l = l[0]
        l.set_linestyle(linestyle[c])
        l.set_marker(markers[c])

    legend = plt.legend(loc="best", shadow=False, frameon=True)
    plt.grid()
    plt.xlabel("Data")
    plt.xlabel("Regularizer")
    plt.title("%s: L-curve" % name)

    try:
        # Open windows (and also save them) in full screen
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
    except:
        pass
    plt.show(block=False)

    if dir_output is not None:
        ph.save_fig(fig, dir_output, "%s_L-curve.pdf" % name)


def show_measures(parameter_study_reader, lines, ctr, dir_output=None):
    name = parameter_study_reader.get_parameter_study_name()

    line_to_parameter_labels_dic = \
        parameter_study_reader.get_line_to_parameter_labels()

    markers = ph.MARKERS * 100

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

        if len(y[0]) < 10:
            markevery = 1
        else:
            markevery = 10

        fig = plt.figure(ph.add_one(ctr))
        fig.clf()
        ax = fig.gca()
        for c in range(len(y)):
            l = plt.plot(y[c], label=labels[c])

            # Extract line object to adjust line settings
            l = l[0]
            l.set_linestyle(linestyle[c])
            l.set_marker(markers[c])
            l.set_markevery(markevery)

        # Only integers on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if m in ["NCC", "SSIM"]:
            ax.set_ylim([0, 1])
        elif m == "NMI":
            ax.set_ylim([1, 1.3])

        legend = plt.legend(loc="best", shadow=False, frameon=True)
        plt.grid()
        plt.xlabel("iteration")
        plt.title("%s: %s" % (name, m))

        try:
            # Open windows (and also save them) in full screen
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        except:
            pass
        plt.show(block=False)

        if dir_output is not None:
            ph.save_fig(fig, dir_output, "%s_%s.pdf" % (name, m))


def show_reconstructions(parameter_study_reader,
                         lines,
                         dir_output=None,
                         colormap="Greys_r",
                         reference=None,
                         reference_mask=None,
                         ):

    try:
        reconstructions_dic = parameter_study_reader.get_reconstructions()
    except IOError as e:
        print("Error: '%s'. Visualization skipped." % e)
        return

    ph.print_info("Prepare comparison of reconstructions ... ")
    name = parameter_study_reader.get_parameter_study_name()
    line_to_parameter_labels_dic = \
        parameter_study_reader.get_line_to_parameter_labels()

    for j, line in enumerate(lines):
        # Get labels
        labels = [line_to_parameter_labels_dic[i] for i in line]

        data_nda = [reconstructions_dic[str(ell)].reshape(
            reconstructions_dic["shape"]) for ell in line]

        if reference is not None:
            data_reader = dr.DataReader(reference)
            data_reader.read_data()
            data_nda.insert(0, data_reader.get_data())
            labels.insert(0, "Reference")

        if len(reconstructions_dic["shape"]) == 2:
            try:
                if len(lines) == 1:
                    filename = name + "_reconstructions.pdf"
                else:
                    filename = name + "_reconstructions_%d.pdf" % (j + 1)

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
                # Convert to float32 (no float16 in SimpleITK)
                recon_sitk = sitk.GetImageFromArray(nda.astype(np.float32))
                recon_sitk.SetSpacing(spacing)
                recon_sitk.SetOrigin(origin)
                recon_sitk.SetDirection(direction)
                recons_sitk.append(recon_sitk)

            labels_sitk = ["%s_%s" % (name, line.replace(".", "p"))
                           for line in labels]

            if reference_mask is not None:
                segmentation_sitk = sitkh.read_nifti_image_sitk(
                    reference_mask, sitk.sitkUInt8)
                try:
                    recon_sitk - sitk.Cast(segmentation_sitk,
                                           recon_sitk.GetPixelIDValue())
                except:
                    raise IOError(
                        "Reference mask must be in same physical as "
                        "the computed reconstructions")
            else:
                segmentation_sitk = None
            sitkh.show_sitk_image(
                recons_sitk,
                label=labels_sitk,
                segmentation=segmentation_sitk,
                dir_output=dir_output,
                show_comparison_file=True)


def main():

    # Read input
    input_parser = InputArgparser.InputArgparser(
        description="Show and analyse stored parameter study.",
    )

    input_parser.add_dir_input(
        help="Input directory where parameter study results are located.",
        required=True)
    input_parser.add_study_name(required=True)
    input_parser.add_dir_output_figures()
    input_parser.add_colormap(default="Greys_r")
    input_parser.add_reference()
    input_parser.add_option(
        option_string="--reference-mask", type=str)
    input_parser.add_option(
        option_string="--show-reconstructions",
        help="Turn on/off visualization of reconstructions",
        type=int,
        default=1,
    )

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

    # Figure ctr
    ctr = [0]
    show_L_curve(parameter_study_reader, lines, ctr, args.dir_output_figures)
    show_measures(parameter_study_reader, lines, ctr, args.dir_output_figures)

    if args.show_reconstructions:
        show_reconstructions(parameter_study_reader,
                             lines,
                             args.dir_output_figures,
                             colormap=args.colormap,
                             reference=args.reference,
                             reference_mask=args.reference_mask)

    return 0

if __name__ == '__main__':
    main()
