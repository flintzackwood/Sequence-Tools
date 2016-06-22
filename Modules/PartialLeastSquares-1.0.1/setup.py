#!/usr/bin/env python

## setup.py

from setuptools import setup, find_packages
import sys, os

setup(name='PartialLeastSquares',
      version='1.0.1',
      author='Avinash Kak',
      author_email='kak@purdue.edu',
      maintainer='Avinash Kak',
      maintainer_email='kak@purdue.edu',
      url='https://engineering.purdue.edu/kak/distPLS/PartialLeastSquares-1.0.1.html',
      download_url='https://engineering.purdue.edu/kak/distPLS/PartialLeastSquares-1.0.1.tar.gz',
      description='A Python module for regression and classification with the Partial Least Squares algorithm',
      long_description=''' 

Version 1.0.1 includes a couple of CSV data files in the
Examples directory that were inadvertently left out of
Version 1.0 packaging of the module.

You may need this module if (1) you are trying to make
multidimensional predictions from multidimensional
observations; (2) the dimensionality of the observation
space is large; and (3) the data you have available for
constructing a prediction model is rather limited.  The more
traditional multiple linear regression (MLR) algorithms are
likely to become numerically unstable under these
conditions.

In addition to presenting the main PLS algorithm that can be
used to make a multidimensional prediction from
multidimensional data, this module also includes what is
known as the PLS1 algorithm for the case when the predicted
entity is just one-dimensional (as in, say, face recognition
in computer vision).

Typical usage syntax:

::

        In typical PLS notation, X denotes the matrix formed by
        multidimensional observation vectors, with each row of X standing
        for the values taken by all the predictor variables.  And Y denotes
        the matrix formed by the multidimensional prediction vectors. Each
        row of Y corresponds to the prediction that can be made on the
        basis of the corresponding row of X.  Let's say that you have the
        observed data for the X and the Y matrices in the form of CSV
        records in disk files. Your goal is to calculate the matrix B of
        regression coefficients with this module.  All you have to do is
        make the following calls:

            import PartialLeastSquares as PLS

            XMatrix_file = "X_data.csv"
            YMatrix_file = "Y_data.csv"

            pls = PLS.PartialLeastSquares(
                    XMatrix_file =  XMatrix_file,
                    YMatrix_file =  YMatrix_file,
                    epsilon      = 0.0001,
                  )
           pls.get_XMatrix_from_csv()
           pls.get_YMatrix_from_csv()
           B = pls.PLS()

        The object B returned by the last call will be a numpy matrix
        consisting of the calculated regression coefficients.  Let's say
        that you now have a matrix Xtest of data for the predictor
        variables.  All you have to do to calculate the values for the
        predicted variables is

           Ytest =  Xtest * B

          ''',

      license='Python Software Foundation License',
      keywords='classification, regression, data dimensionality reduction',
      platforms='All platforms',
      classifiers=['Topic :: Scientific/Engineering :: Information Analysis', 'Programming Language :: Python :: 2.7', 'Programming Language :: Python :: 3.2'],
      packages=[]
)
