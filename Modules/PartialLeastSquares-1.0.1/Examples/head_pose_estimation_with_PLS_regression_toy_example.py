#!/usr/bin/env python

## head_pose_estimation_with_PLS_regression_toy_example.py

##  IMPORTANT: The goal of this script is ONLY to show how you can use the
##             module for head pose estimation with PSL based regression In and
##             of itself, this script is NOT to be taken as a serious
##             investigation of the head pose estimation problem.

##  ASSUMPTIONS:
##  
##    This method assumes that the image directory contains two subdirectories named:
##    
##        --  training
##    
##        --  testing
##
##    Furthermore, the script assumes that the name of each image file in the two
##    subdirectories named above is an encoding of the roll, pitch, and yaw values
##    associated with the face image in that image.  For example, the name of the
##    first image file in the directory `/head_pose_images/training/' is
##
##            y1p1r2.jpg
##    
##    This name implies that the pose of the head in this image corresponds to the
##    following values for roll, pitch, and yaw:
##    
##            yaw   = -30 degrees
##            pitch = -30 degrees
##            roll  = -20 degrees
##    
##    To understand why the name of the file translates into the values shown
##    above, note that the pose of the head is varied with respect to each of the
##    roll, pitch, and yaw parameters from -30 degrees to +30 degrees. We use the
##    following mapping between the integer indices associated with the paramters
##    in the file names and their actual angles:
##    
##           1   =>    -30 deg
##           2   =>    -20 deg
##           3   =>    -10 deg
##           4   =>     0  deg
##           5   =>    +10 deg
##           6   =>    +20 deg
##           7   =>    +30 deg
##    
##     This naming convention makes it easy to to create the rows of the Y matrix
##     for each row of the X matrix.  Each row of the X matrix is the vectorized
##     representation of the pixels in the image and each corresponding row of the
##     Y matrix consists of the three pose angles associated with that image.
##    '''

import PartialLeastSquares as PLS

image_directory = 'head_pose_images'

pls = PLS.PartialLeastSquares(
            epsilon      = 0.001,
            image_directory = image_directory,
            image_type = 'jpg',                    
            image_size_for_computations = (40,40), 
      )

pls.vectorize_images_and_construct_X_and_Y_matrices_for_head_pose_estimation_with_PLS()
B = pls.PLS()
print("\nThe regression matrix B is of size: %d rows and %d columns\n" % B.shape)
print("Displaying the matrix B of regression coefficients. (By default,\n" + \
      "numpy shows only a small portion of a large matrix):\n")
print(B)
pls.run_evaluation_of_PLS_regression_for_head_pose_estimation()


