#!/usr/bin/env python

## PLS_regression_with_supplied_X_and_Y.py

import PartialLeastSquares as PLS


XMatrix_file = "X_data.csv"
YMatrix_file = "Y_data.csv"

pls = PLS.PartialLeastSquares(
           XMatrix_file =  XMatrix_file,
           YMatrix_file =  YMatrix_file,
           epsilon      = 0.001,
      )
pls.get_XMatrix_from_csv()
pls.get_YMatrix_from_csv()
B = pls.PLS()
print("\nDisplaying the matrix of regression coefficients:")
print(B)
print("\n")
pls.apply_regression_matrix_interactively_to_one_row_of_X_to_get_one_row_of_Y()
