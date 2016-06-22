import PartialLeastSquares as PLS
import unittest

XMatrix_file = "X_data.csv"
YMatrix_file = "Y_data.csv"


class TestRegressionMatrixCalculation(unittest.TestCase):

    def setUp(self):
        print("Testing PLS regression on sample X and Y matrices")
        self.pls = PLS.PartialLeastSquares(XMatrix_file =  "X_data.csv",
                                           YMatrix_file =  "Y_data.csv")
        self.pls.get_XMatrix_from_csv()
        self.pls.get_YMatrix_from_csv()
        B = self.pls.PLS()
        print("\nDiplaying the matrix of regression coefficients:\n")
        print(B)
        print("\n\n")

    def test_regression_matrix_calculation(self):
        B = self.pls.PLS()
        self.assertGreater(B[0,0], -1.7) and self.assertLess(B[0,0], -1.65)
        self.assertGreater(B[1,1], 0.28) and self.assertLess(B[1,1], 0.3)
        self.assertGreater(B[2,2], 0.45) and self.assertLess(B[2,2], 0.55)

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestRegressionMatrixCalculation, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

