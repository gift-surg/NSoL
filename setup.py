###
# \file setup.py
#
# Install with symlink: 'pip install -e .'
# Changes to the source file will be immediately available to other users
# of the package
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#
# \see https://python-packaging.readthedocs.io/en/latest/minimal.html
# \see https://python-packaging-user-guide.readthedocs.io/tutorials/distributing-packages/


from setuptools import setup

description = 'Numerical Solver Library to estimate argmin_x [f(x) + alpha g(x)]'
long_description = "This library contains the implementation of several " \
    "numerical solvers to estimate a solution for " \
    "argmin_x [f(x) + alpha g(x)]. " \
    "Provided solver include Tikhonov, ADMM and Primal-Dual solvers."

setup(name='NSoL',
      version='0.1.dev1',
      description=description,
      long_description=long_description,
      url='https://cmiclab.cs.ucl.ac.uk/gift-surg/NSoL',
      author='Michael Ebner',
      author_email='michael.ebner.14@ucl.ac.uk',
      license='BSD-3-Clause',
      packages=['nsol'],
      install_requires=['pysitk'],
      zip_safe=False,
      keywords='development numericalsolver convexoptimisation',
      classifiers=[
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
      ],

      )
