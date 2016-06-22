#!/usr/bin/env python

import unittest
import TestRegressionMatrixCalculation
import TestInstallationOfImageDatasets

class PartialLeastSquaresTestCase( unittest.TestCase ):
    def checkVersion(self):
        import PartialLeastSquares

testSuites = [unittest.makeSuite(PartialLeastSquaresTestCase, 'test')] 

for test_type in [
            TestRegressionMatrixCalculation,
            TestInstallationOfImageDatasets
    ]:
    testSuites.append(test_type.getTestSuites('test'))


def getTestDirectory():
    try:
        return os.path.abspath(os.path.dirname(__file__))
    except:
        return '.'

import os
os.chdir(getTestDirectory())

runner = unittest.TextTestRunner()
runner.run(unittest.TestSuite(testSuites))
