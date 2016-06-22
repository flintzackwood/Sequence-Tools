#!/usr/bin/env python

## face_recognition_with_PLS_toy_example.py

##  IMPORTANT: The goal of this script is ONLY to show how you can use the
##             module for face recogniton work.  In and of itself, this
##             script is NOT to be taken as a serious investigation of face
##             recognition with PLS.


## ASSUMPTIONS:
##
##  This method assumes that the images to be used for training and testing are
##  organized as follows in the image_directory option supplied to the
##  constructor of the module:
##
##                                  image_directory
##                                       |
##                                       |
##                     --------------------------------------
##                    |                                      |
##                    |                                      |
##                 training                              testing
##                    |                                      |
##                    |                                      |
##        -------------------------             -----------------------------
##       |                         |           |                             |
##       |                         |           |                             |
##   positives                 negatives    positives                    negatives
##
##
##  The module constructs the X and the Y matrices from the images in the
##   `training/positives' and the `training/negatives' subdirectories.  The
##   vectorized representation of each image constitutes a row of the X
##   matrix. The corresponding element in the one-column Y matrix is +1 for the
##   images in the `positives' directory and -1 for the images in the `negatives'
##   directory.  In a similar manner, the method constructs Xtest and Ytest
##   matrices from the images in the `testing/positives' and `testing/negatives'
##   subdirectories.


import PartialLeastSquares as PLS

image_directory = 'face_images'

pls = PLS.PartialLeastSquares(
            epsilon      = 0.001,
            image_directory = image_directory,
            image_type = 'jpg',                    
            image_size_for_computations = (40,40), 
      )
pls.vectorize_images_and_construct_X_and_Y_matrices_for_face_recognition_with_PLS1()
pls.PLS1()
pls.run_evaluation_of_PLS_regression_for_face_recognition()

