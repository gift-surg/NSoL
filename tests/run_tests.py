#!/usr/bin/python

##
# \file run_tests.py
# \brief      main-file to run specified unit tests
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#


# Import libraries
import unittest
import sys
import os

# Import modules for unit testing
# from kernels_test import *
# from loss_functions_test import *
from similarity_measures_test import *
# from solvers_test import *

if __name__ == '__main__':
    print("\nUnit tests:\n--------------")
    unittest.main()
