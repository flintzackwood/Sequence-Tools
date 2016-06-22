#!/usr/bin/env python

__version__ = '1.0.1'
__author__  = "Avinash Kak (kak@purdue.edu)"
__date__    = '2014-July-29'
__url__     = 'https://engineering.purdue.edu/kak/distPLS/PartialLeastSquares-1.0.1.html'
__copyright__ = "(C) 2014 Avinash Kak. Python Software Foundation."

__doc__ = '''

    PartialLeastSquares.py

    Version: ''' + __version__ + '''
   
    Author: Avinash Kak (kak@purdue.edu)

    Date: ''' + __date__ + '''


    @title
    CHANGE LOG:

        Version 1.0.1 includes a couple of CSV data files in the Examples
        directory that were inadvertently left out of Version 1.0 packaging
        of the module.  The module code remains unchanged.


    @title
    INTRODUCTION:

        You may need this module if (1) you are trying to make
        multidimensional predictions from multidimensional observations;
        (2) the dimensionality of the observation space is large; and (3)
        the data you have available for constructing a prediction model is
        rather limited.  The more traditional multiple linear regression
        (MLR) algorithms are likely to become numerically unstable under
        these conditions.  When the dimensionality of the space defined by
        your predictor variables is large and the number of observations you
        have available for constructing a prediction model is small, the
        traditional regression algorithms are likely to suffer from the
        "multicollinearity problem" caused by strong correlations between
        the different predictor variables. (We can think of each dimension
        of the observation space as corresponding to one predictor
        variable.)  In the rest of this Introduction, let's denote the
        predictor variables by x1, x2, ..., xN and the predicted variables
        by y1, y2, y3, ..., yM.  So N is the dimensionality of the space
        defined by the predictor variables and M the dimensionality of the
        space defined by the predicted variables.

        With the traditional multiple linear regression algorithms, one
        typically gets around the multicollinearity problem by first
        reducing the dimensionality N of the space of the predictor
        variables through the application of Principal Components Analysis
        (PCA) to your data. And another way of getting around the same
        problem is by using the method of Partial Least Squares (PLS) that
        this module implements.  PCA focuses solely on the space defined by
        the predictor variables and gives you a small orthogonal set of
        directions in that space that explain the most significant
        variations in just the space defined by x1, x2, ...., xN.
        Subsequently, you can use the principal directions (these will be
        linear combinations of the original predictor variables) for making
        predictions.  PLS also gives you a small set of principal
        directions (also called principal components) in the space defined
        by the original predictor variables.  However, the directions
        yielded by PLS also take into account the observed variations in the
        space defined by the predicted variables y1, y2, ...., yM.  So if a
        principal component cannot explain the variations in the space
        defined by the predicted variables y1, y2, ..., yM, it will be
        ignored by PLS.

        Another way of saying the same thing is that whereas the principal
        components returned by PCA explain the largest variances in the
        space of the predictor variables, the principal components in the
        same space that are returned by PLS depend on their power to
        predict the variances in the space defined by the predicted
        variables.  Therefore, if it should happen that a principal
        component in the space of the predictor variables cannot be used to
        explain the the observed variability in the space defined by the
        predicted variables, PLS will ignore that principal component.  In
        the context of PLS, the principal components are more commonly
        referred to as latent vectors.

        If you have previously used multiple linear regression for data
        prediction and this happens to be your first exposure to PLS for
        making multidimensional predictions, you are likely to ask the
        following question: Why not consider each predicted variable
        separately in its dependence on the predictor variables? (In the
        context of linear regression, the predicted variables are
        frequently called the dependent or the response variables and the
        predictor variables as the explanatory variables.)  To answer this
        question, yes, you can certainly do that if the data you are using
        does not suffer from the problems listed at the beginning of this
        Introduction.  However, if those problems exist, you'll have no
        choice but to apply PCA to the explanatory variables in order to
        reduce the dimensionality of the space defined by those variables.
        And, once you have taken the leap to dimensionality reduction, you
        might as well go all the way and use PLS for reasons already
        mentioned.


    @title
    SOME EXAMPLE PROBLEM DOMAINS FOR PLS:

        Consider the problem of head pose estimation in computer vision.
        Let's say you want to associate three pose parameters --- roll,
        pitch, and yaw --- with the pose of the face seen in a mostly
        frontal image of an individual as captured by a surveillance
        camera.  In a regression based approach to solving this problem,
        you'll first create a dataset of the images of the face for
        different value for the three pose parameters. [These days, such a
        dataset can be created by the computer itself from a single RGBD
        image of of the individual as recorded by the Microsoft Kinect
        sensor.  On account of the depth information that is also captured
        by this sensor, the computer can apply pose transformations to the
        recorded RGBD image and generate 2D images for the different values
        of roll, pitch, and yaw.]  Let's say each 2D face image is
        represented by a 40x40 array of pixels. If you represent each image
        as a vector of 1600 pixel values, you'll have 1600 predictor
        variables for the pixels values and 3 predicted variables for the
        roll, pitch, and the yaw parameters of the head pose.  The PLS
        algorithm when applied to this dataset will return a matrix of
        regression coefficients.  Subsequently, by applying this matrix to
        any vectorized representation of a face image, you will be able
        estimate the head pose associated with the face in that image.  The
        Examples directory of this module includes a script for doing
        exactly this.

        For another example, let's say you want to predict the chemical
        composition of a compound through spectroscopy. Each observation
        here would consist of the values shown by a spectrograph at, say, N
        hundred different frequencies and the predicted output would be the
        proportion of the more elemental types in the compound.  If you
        expect to see M elemental types in a compound, we are talking about
        N predictor variables and M predicted variables.

        For yet another example, let's say that our assessment of the
        physical health of an individual depends on a number of parameters
        such as blood-pressure, cholesterol, blood glucose, triglycerides,
        etc.  And let's further say that we want to predict these
        parameters simultaneously from a multidimensional input consisting
        of average daily fat intake, the amount of sleep, the length of
        time exercising, etc.  


    @title
    THE NOTATION OF PLS:

        In the context of PLS, the matrix formed by the recorded values for
        the predictor variables x1, x2, ..., xN is typically denoted X.
        Each row of X is one observation record and each column of X stands
        for one predictor variable.  The values for the predicted variables
        y1, y2, ..., yM are also placed in a matrix that is typically
        denoted Y.  Each column of Y stands for one predicted variable and
        each row of Y for the values for all the predicted variables using
        the corresponding row of X for prediction.

        So if we have I observation records available to us, X is a matrix
        of size IxN and Y is a matrix of size IxM.

        Each principal component of X, usually referred to as a latent
        vector of X, is represented by t and the matrix whose columns are
        the latent vectors t is typically denoted T. If the p is the number
        of latent vectors discovered for X, then T is of size Ixp.  Along
        the same lines, each latent vector of Y is typically denoted u and
        the matrix formed by these latent vectors is denoted U. In most
        variants of the PLS algorithm, the latent vectors for X and Y are
        discovered conjointly.  In such cases, U will also be of size Ixp.

        Each latent vector t is a weighted linear combination of the
        columns of X and each latent vector u is a weighted linear
        combination of the columns of Y.  The weights that go into these
        combinations can also be thought of as vectors. These weights are
        referred to as loadings. The loading vectors for X are represented
        by p and those for Y by q.  The matrix formed by the p vectors
        represented by P and by the q vectors by Q.

        Two more matrices important to the discovery of the latent vectors
        for X and Y are W and C.  As mentioned earlier, a latent vector t
        is merely a weighted linear combination of the columns of X. For a
        calculated t, we represent these weights in the form of a vector w,
        and all the w vectors (for the different t vectors) taken together
        constitute the matrix W.  By the same token, a latent vector u is
        is a weighted linear combination of the columns of Y and, for a
        given u, we represent these weights in the form of a vector c.  All
        the different vectors c (for the different u vectors) constitute
        the matrix C.

        Finally, we need a matrix B of regression coefficients.  In fact,
        figuring out what B should be is the main purpose of the PLS
        algorithm.  Once we have B, given a test data matrix Xtest, we can
        make the prediction Ytest = (Xtest * B).  For a given pair of latent
        vectors discovered simultaneously, t from X and u from Y, the
        product (t' * u) tells as to what extent the t portion of X can
        predict the u portion of Y.  We denote this product by the scalar
        variable b.

    @title
    ALGORITHMIC DETAILS:

        The PLS algorithm was developed originally by Hermon Wold in 1966
        and, over the years, there have come into existence several
        variants of the original algorithm. The different versions of the
        algorithm differ in the following ways: (1) As each pair of latent
        vectors, t from X and u from Y, is discovered, the different
        versions of PLS differ with regard to the deflation step. Recall
        that by deflation we mean how we subtract the influence of t from X
        and of u of Y before starting search for the next pair (t,u) of
        latent vectors. (2) The different versions different with regard to
        the normalization of the latent vectors. And (3) the different
        versions differ with regard to how the matrix B of the regression
        coefficients is calculated.  

        See the review paper "Overview and Recent Advances in Partial Least
        Squares" by Roman Rosipal and Nicole Kramer, LNCS, 2006, for
        additional information related to the different variants of PLS and
        for citations to more recent work related to the algorithm.

        This module implements three variants of the PLS algorithm.  We
        refer to them as PLS(), PLS1(), and PLS2().  The implementation of
        PLS() is based on the description of the algorithm by Herve Abdi in
        the article "Partial Least Squares Regression and Projection on
        Latent Structure Regression," Computational Statistics, 2010.  From
        my experiments with the different variants of PLS, this particular
        version generates the best regression results.  The Examples
        directory contains a script that carries out head-pose estimation
        using this version of PLS.

        The variants PLS1() and PLS2() are based on the description of the
        algorithm in the previously cited paper by Roman Rosipal and Nicole
        Kramer.  The PLS1() algorithm has a special role amongst all of the
        variants of the partial least squares method --- it is meant
        specifically for the case when the matrix Y consists of only one
        column vector.  That is, we use PLS1() when there is just one
        predictor variable.  As you will see from the code in the Examples
        directory, this makes PLS1() particularly appropriate for solving
        recognition problems in computer vision.

        The rest of this section presents the main steps of a PLS
        algorithm.  As should be clear to the reader by this time, our main
        goal is to find the latent vector t in the column space of X and
        the latent vector u in the column space of Y so that the covariance
        (t' * u)/I is maximized.  As mentioned earlier, t' is the transpose
        of t and I is the number of rows in X (and in Y).  Starting with a
        random guess for u, we iterate through the following eights steps
        until achieving the termination condition stated in the last step:

         1)  w = X' * u

                  where X' is the transpose of X and u is the current value
                  of the latent vector for Y.  We will use the elements of
                  the vector w thus obtained as weights for creating a
                  linear combination of the columns of X as a new candidate
                  for t.

         2)  w =  w / ||w||

                  where ||w|| is the norm of the vector w.  That is, ||w||
                  = sqrt(w' * w).  This step normalizes the magnitude of w
                  to 1.

         3)  t = X * w

                  Now we have a new approximation to t.  For PLS, we also
                  normalize t as we normalized w in the previous step.

         4)  c = Y' * t

                  We will use the elements of the vector c thus obtained as
                  weights for creating a linear combination of the columns
                  of Y for a new candidate as Y's latent vector u.

         5)  c = c / ||c||

                  We normalize c just as we normalized w.


         6)  u_old  =  u

                  We store away the currently used value for u

         7)  u = Y * c

                  Now we have a new candidate for the latent vector for Y.

         8)  terminate the iterations if 

                           ||u - u_old||  <  epsilon
       
             where epsilon is a user-specified value.

        Once we have a vector t from the column space of X and a vector u
        from the column space of Y, we need to figure out what weighted
        linear combination of the columns of X would result t and what
        weighted linear combination of the column vectors of Y would result
        in Y.  Referring to these weights by the vectors p and q, we
        calculate

             p  =  (X' * t) / ||t||

             q  =  (Y' * u) / ||u||

        If this is our first pair (t,u) of latent vectors, we initialize
        the matrices T and U by setting them to the column vectors t and u,
        respectively.  If this is not the first pair (t,u), we augment T
        and U by the additional column vectors t and u, respectively.  The
        same goes for the vectors p and q vis-a-vis the matrices P and Q.

        After the calculation of each pair (t,u) of latent vectors, it's
        time to subtract out from X the contribution made to it by t, and
        to subtract out from Y the contribution made to it by u. This step,
        called deflation as mentioned previously, is implemented as
        follows for PLS1 and PLS2:

             X  =  X - (t * p')

             Y  =  Y - (t * t' * Y) / ||t||

        For PLS, the deflation is carried out by

             X  =  X - (t * p')

             Y  =  Y - b * t * c'

        where b = t' * u. 

        This process of first finding the covariance maximizing latent
        vectors t and u from the column spaces of X and Y, respectively,
        and subsequently deflating X and Y as shown above is continued
        until there is not much left of the matrices X and Y.  What is left
        of X and Y after each such deflation can be measured by taking the
        Frobenius norm of the matrices, which is simply the square-root of
        the sum of the squares of all the matrix elements.

        After the latent vectors of X and Y have been extracted, the last
        step is the calculation of the matrix B of regression
        coefficients. For the PLS() implementation, this calculation
        involves

                     _                  
            B =  (P')   *  diag[b1, b2, ...., bp] * C'

                   _
        where  (P')   is the pseudo-inverse of the transpose of P. The
        second term above is the diagonal matrix formed by the b values
        computed after the calculation of each pair (t,u) as mentioned
        earlier. C' is the transpose of the C matrix.

        For PLS1() and PLS2(), the B matrix is calculated through the
        following formula:

            B = W * (P' * W)^-1 * C'

        where (P' * W)^-1 is the matrix inverse of the product of the two
        matrices P' and W.  Note that P' is the transpose of P.

        After you have calculated the B matrix, you are ready to make
        predictions in the future as new row vectors for the X matrix come
        along.  Let's use the notation Xtest to denote the new data
        consisting of the observed values for the predictor variables.
        Your prediction would now consist of

            Ytest =  Xtest * B


    @title
    USAGE:

        Let's say that you have the observed data for the X and the Y
        matrices in the form of CSV records in disk files. Your goal is to
        calculate the matrix B of regression coefficients with this module.
        All you have to do is make the following calls:

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

        Obviously, Xtest may contain just a single row --- meaning that it
        may contain a single observation consisting of the values for all
        the predictor variables.  In that case, the computed Ytest will
        also be just a one-row matrix.

        The module can also work directly off the image data.  Going back
        to the problem of head-pose estimation on the basis of near-frontal
        image of a human head, let's say that your training data consisting
        of 2D face images with known values for the roll, pitch, and yaw
        parameters of the individual's head is in a directory called
        `head_pose_images'.  The following calls will create a PLS based
        regression framework for this case:

            import PartialLeastSquares as PLS

            image_directory = 'head_pose_images'

            pls = PLS.PartialLeastSquares(
                        epsilon      = 0.001,
                        image_directory = image_directory,
                        image_type = 'jpg',
                        image_size_for_computations = (40,40),
                  )
            pls.vectorize_images_and_construct_X_and_Y_matrices_for_PLS2()
            B = pls.PLS()
            pls.run_evaluation_of_PLS_regression_with_test_data(B)

        where the object B returned by the next to the last call is the
        matrix of regression coefficients.  The module expects the image
        directory to consist of two sub-directories called `training' and
        `testing'.  The PLS() is applied only to the image data in the
        `training' subdirectory. The method

             run_evaluation_of_PLS_regression_with_test_data(B)

        in the last call applies the matrix B of regression coefficients to
        the Xtest matrix constructed from the 2D face images stored in the
        `testing' subdirectory of the image directory.  The README in the
        Examples directory has further information on how we evaluate the
        accuracy of PLS based regression.

        If your goal is to carry out image recognition, as in face
        recognition, with PLS, you'd want to use the PLS1() implementation.
        Assuming that your training face images are in a directory called
        `face_images', your call to the module will look like:
        
            import PartialLeastSquares as PLS

            image_directory = 'face_images'

            pls = PLS.PartialLeastSquares(
                        epsilon      = 0.001,
                        image_directory = image_directory,
                        image_type = 'jpg',
                        image_size_for_computations = (40,40),
                  )

            pls.vectorize_images_and_construct_X_and_Y_matrices_for_PLS1()
            pls.PLS1()
            pls.run_evaluation_with_positive_and_negative_test_image_data()

        When called for image recognition as shown above, the module
        expects the image_directory to contain two subdirectories called
        `training' and `testing'.  The module also expects that each of
        these two subdirectories will contain two subdirectories called
        `positives' and `negatives'. The module constructs the X matrix
        from vectorized representation of the images in the
        `training/positives' and `training/negatives' subdirectories.  For
        the images in the former, it places a +1 label in the corresponding
        row of a one-column Y matrix.  And for each image in the latter, it
        enters a -1 in the same Y matrix.  The regression matrix B
        constructed from X and Y constructed in this manner is is applies
        to the Xtest matrix constructed from the images in the
        'testing/positives' and `testing/negatives' subdirectories.
    

    @title
    CONSTRUCTOR PARAMETERS:

        XMatrix_file :       The name of a CSV file that contains the X matrix.
                             Each line of this file should contain the entries
                             for one row of the X matrix.

        YMatrix_file :       The name of a CSV file that contains the Y matrix.
                             Each line of this file should contain the entries
                             for one row of the Y matrix.

        epsilon:             Required for the testing the termination condition
                             for PLS iterations.  Search for latent vectors is
                             stopped when the Frobenius norm of what is left
                             of the X matrix is less than the value of epsilon.
                             If left unspecified, it defaults to 0.0001.

        image_directory:     This is the name of the directory that has the images for
                             experimenting with PLS regression for head-pose
                             estimation and face recognition. How the images are
                             distributed between the training and the testing 
                             subdirectories depends on whether they are meant to
                             be used for head-pose estimation or for binary 
                             recognition.

        image_type:          Specify the image type (such as jpg, png, etc.) 
                             in the image_directory

        image_size_for_computations:  Specify the array size to be used for 
                                      image computations.  If the actual images 
                                      larger, the module will reduce them to 
                                      this size.  On the other hand, if the 
                                      actual images are smaller along either or
                                      both dimensions, the module will zero-pad
                                      them appropriately to bring the size up 
                                      to the values specified by this option. 


    @title
    METHODS:

        (1) get_XMatrix_from_csv()

            If you wish to use your own X and Y matrices for PLS regression, you'd
            need to supply them in the form of CSV files.  This method extracts the X matrix
            from the file named for this purpose by the constructor option XMatrix_file.

        (2) get_YMatrix_from_csv()
 
            The comment made above applies here also. This method extracts the Y matrix
            from the file named for this purpose by the constructor option YMatrix_file.

        (3) vectorize_images_and_construct_X_and_Y_matrices_for_head_pose_estimation_with_PLS()

            Assuming that you supplied the constructor with the `image_directory'
            option, this method further assumes that the images to be used for PLS
            regression are in two subdirectories named `training' and `testing'. The
            method also assumes that the values of the predicted variables for each
            image file are encoded into the names of the files themselves.  See the
            docstring associated with this method for more information.

        (4) vectorize_images_and_construct_X_and_Y_matrices_for_face_recognition_with_PLS1()

            This method assumes the directory structure shown below for the images to be
            used for face recognition:

                                                                -------  positives
                                                               |
                                       ------  training ------ |
                                      |                        |
                                      |                         -------  negatives
               image_directory  ------|                        
                                      |                         -------  positives
                                      |                        |
                                       ------  testing  -------|
                                                               |
                                                                ------- negatives

            The method constructs the X and Y matrices needed for PLS regression
            from the vectorized representation of the images in the
            `training/positives' and `training/negatives' directories. See the
            docstring for this method for additional information.


        (5) run_evaluation_of_PLS_regression_for_head_pose_estimation()

            The method here uses the Xtest and Yest matrices constructed by the
            vectorize_images_and_construct_X_and_Y_matrices_for_head_pose_estimation_with_PLS()
            method from the images in the `testing' subdirectory of the
            image_directory option supplied to the constructor.  The Xtest and Ytest
            matrices are then used for evaluating PLS regression for head pose
            estimation.

        (6) run_evaluation_of_PLS_regression_for_face_recognition()

            The method here uses the Xtest and Yest matrices constructed by the
            vectorize_images_and_construct_X_and_Y_matrices_for_face_recognition_with_PLS1()
            method from the images in the `testing/positives' and the
            /testing/negatives/ subdirectories for evaluating PLS regression for face
            recognition.

        (7) PLS()

            This implementation of PLS is based on the description of the algorithm
            by Herve Abdi in the article "Partial Least Squares Regression and
            Projection on Latent Structure Regression," Computational Statistics,
            2010.  From my experiments with the different variants of PLS, this
            particular version generates the best regression results.  The Examples
            directory contains a script that carries out head-pose estimation using
            this version of PLS.

        (8) PLS1()

            This implementation is based on the description of the algorithm in the
            article "Overview and Recent Advances in Partial Least Squares" by Roman
            Rosipal and Nicole Kramer, LNCS, 2006.  PLS1 assumes that the Y matrix
            consists of just one column. That makes it particularly appropriate for
            solving face recognition problems.  This module uses this method for a
            two-class discrimination between the faces.

        (9) PLS2()

            This implementation is based on the description of the algorithm as the
            same source as for PLS1().


    @title
    THE EXAMPLES DIRECTORY:

        The best way to become familiar with this module is by executing the
        following scripts in the Examples subdirectory:

        1.  PLS_regression_with_supplied_X_and_Y.py

                This script computes the matrix B of regression coefficients from the
                X and Y matrices you supply in the form of CSV files through two
                constructor options.  First run the script with the two CSV files
                that are already in the Examples directory.  Subsequently, you can
                use your own CSV files for the X and Y matrices.
                
        2.  head_pose_estimation_with_PLS_regression_toy_example.py

                This script first learns the regression-coefficients matrix B from
                the images in a `training' directory and then applies the regression
                to estimate the head pose for the 2D images in a `testing' directory.
                For the purpose of training and testing, the script assumes that the
                name of each image file encodes the associated values of the roll,
                pitch, and yaw parameters of the head pose in the image.

        3.  face_recognition_with_PLS_toy_example.py

                This script uses the PLS1 algorithm for face recognition.  The PLS1
                algorithm is designed specifically for the case when the Y matrix
                consists of only one column.  For each row of the X matrix (which is
                a vector representation of a face), we place in the Y matrix +1 for
                the positive faces and -1 for the negative faces.  This is
                done separately for the training and the testing images.
               

    @title  
    INSTALLATION:

        After you have downloaded the compressed archive into any directory of your
        choice, do the following:

        1) Uncompress the archive with a tool of your choice.  In a Linux environment,
           you are likely to use the following command:

                 tar zxvf  the_downloaded_compressed_archive

        2) Next cd into the Examples subdirectory of the installation directory:

                 cd  installation_directory/Examples

           where `installation_directory' is your pathname to the directory where the
           top-level uncompressed files and subdirectories reside.

        3) In the Examples directory, do the following

                 tar zxvf face_images_toy_dataset.gz
 
                 tar zxvf head_pose_images_toy_dataset.gz

        4) Now go back into the installation directory and execute the setup.py
           script:

                 sudo python setup.py install                 

        You have to have root privileges for this to work.  On Linux distributions,
        this will install the module file at a location that looks like

             /usr/lib/python2.7/dist-packages/

        If you do not have root access, you have the option of working
        directly off the directory in which you downloaded the software by
        simply placing the following statements at the top of your scripts
        that use the PartialLeastSquares module:

            import sys
            sys.path.append( "pathname_to_PartialLeastSquares_directory" )

        To uninstall the module, simply delete the source directory, locate where
        PartialLeastSquares was installed with "locate partialleastsquares" and
        delete those files.  As mentioned above, the full pathname to the installed
        version is likely to look like
        /usr/lib/python2.7/dist-packages/PartialLeastSquares*

        If you want to carry out a non-standard install of the PartialLeastSquares
        module, look up the on-line information on Disutils by pointing your browser
        to

              http://docs.python.org/dist/dist.html


        The PartialLeastSquares modules was packaged using setuptools.  

    @title
    ACKNOWLEDGMENTS

    The image dataset in the `head_pose_images' subdirectory of the Examples
    directory was created by Dave Kim of the Robot Vision Lab at Purdue. Thanks,
    Dave!

    @title     
    BUGS:
                                 
    Please notify the author if you encounter any bugs.  When sending email,
    please place the string 'PLS' in the subject line.


    @title
    ABOUT THE AUTHOR:  

        Avi Kak has just finished the last book of his three-volume Objects Trilogy
        project.  See his web page at Purdue for what this project is all about.

'''

from PIL import Image
import numpy
import numpy.linalg
import re
import sys, os
import functools
import glob

numpy.set_printoptions(precision=3)

def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value

#----------------------------- PartialLeastSquares Class Definition --------------------------------

class PartialLeastSquares(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''constructor can only be called with keyword arguments for 
                      the following keywords: XMatrix_file, YMatrix_file, epsilon, 
                      image_directory, image_type, image_size_for_computations, debug''')       
        XMatrix_file=YMatrix_file=epsilon=image_directory=image_type=image_size_for_computations=debug=None
        if 'XMatrix_file' in kwargs     :                    XMatrix_file=kwargs.pop('XMatrix_file')
        if 'YMatrix_file' in kwargs     :                    YMatrix_file=kwargs.pop('YMatrix_file')
        if 'epsilon' in kwargs          :                         epsilon=kwargs.pop('epsilon') 
        if 'image_directory' in kwargs  :                 image_directory=kwargs.pop('image_directory') 
        if 'image_type' in kwargs       :                      image_type=kwargs.pop('image_type') 
        if 'debug' in kwargs            :                           debug=kwargs.pop('debug') 
        if 'image_size_for_computations' in kwargs :    
                               image_size_for_computations=kwargs.pop('image_size_for_computations') 
        if len(kwargs) != 0:
                           raise ValueError('''You have provided unrecognizable keyword args''')
        if XMatrix_file: 
            self.XMatrix_file = XMatrix_file
        if YMatrix_file: 
            self.YMatrix_file =  YMatrix_file
        if epsilon:
            self.epsilon = epsilon
        else:
            self.epsilon = .0001           
        if image_directory:
            self.image_directory = image_directory
        if image_type:
            self.image_type = image_type
        if image_size_for_computations:
            self.image_size_for_computations = image_size_for_computations
        if debug:
            self.debug = debug
        else:
            self.debug = 0
        self.X = None                                # Each column of X stands for a predictor variable
        self.Y = None                                # Each column of Y stands for a predicted variable
        self.mean0X = None                           # Store column-wise mean for X
        self.mean0Y = None                           #    and for Y
        self.Xtest = None                            # X matrix for evaluating PLS regression
        self.Ytest = None                            # Y matrix for evaluating PLS regression
        self.B = None                                # regression coefficients
        self.training_positives = []                 # list of row vectors
        self.training_negatives = []
        self.testing_positives = []
        self.testing_negatives = []
        self.testing_positives_filenames = []
        self.testing_negatives_filenames = [] 


    def get_XMatrix_from_csv(self):
        """
        If you wish to use your own X and Y matrices for PLS regression, you'd need
        to supply them in the form of CSV files.  This method extracts the X
        matrix from the file named for this purpose by the constructor option
        XMatrix_file.
        """
        self.X = self._get_matrix_from_csv_file(self.XMatrix_file)
        self.N = self.X.shape[0]
        self.num_predictor_vars = self.X.shape[1]
        print("\n\nThe X matrix: ")
        print(self.X)

    def get_YMatrix_from_csv(self):
        """
        If you wish to use your own X and Y matrices for PLS regression, you'd need
        to supply them in the form of CSV files.  This method extracts the Y
        matrix from the file named for this purpose by the constructor option
        YMatrix_file.
        """
        self.Y = self._get_matrix_from_csv_file(self.YMatrix_file)
        if (self.Y.shape[0] != self.N):
            sys.exit("The X and Y matrix data are not consistent")
        print("\n\nThe Y matrix: ")
        print(self.Y)

    def _get_matrix_from_csv_file(self, filename):
        if not filename.endswith('.csv'): 
            sys.exit("Aborted. get_training_data_from_csv() is only for CSV files")
        all_data = [line.rstrip().split(',') for line in open(filename,"rU")]
        num_rows = len(all_data)
        num_cols = len(all_data[0])
        if self.debug:
            print("num rows: " + str(num_rows) + "  num columns: " + str(num_cols))
        all_data = [[convert(entry) for entry in all_data[i]] for i in range(len(all_data))]
        if self.debug:
            print(all_data)
        matrix = numpy.matrix(all_data)
        if self.debug:
            print(matrix)
        return matrix

    def apply_regression_matrix_interactively_to_one_row_of_X_to_get_one_row_of_Y(self):
        first_message = "\n\nEnter your values for the predictor variables.\n" +  \
                        "The numbers you enter must be space separated.\n" +      \
                        "You need to enter as many numbers as the number of\n" +  \
                        "columns in the X matrix used for calculating B.\n\n" +   \
                        "For starters, you could enter a row of the X used\n" +   \
                        "for calculating B: "
        if sys.version_info[0] == 3:
            answer = input("\n\nWould you like to apply this regression matrix to new\n" + \
                           "data.  Enter `y' for yes or `n' for no: ")
        else:
            answer = raw_input("\n\nWould you like to apply this regression matrix to new\n" + \
                           "data.  Enter `y' for yes or `n' for no: ")
        if answer == 'n': 
            sys.exit(0)
        else:
            first_try = 1
            while 1:
                if first_try:
                    if sys.version_info[0] == 3:
                        new_data = input(first_message)
                    else:
                        new_data = raw_input(first_message)
                else:
                    if sys.version_info[0] == 3:            
                        new_data = input("\n\nEnter another set of values for the predictor\n" + \
                                         "variables, or `n' to quit: ")
                    else: 
                        new_data = raw_input("\n\nEnter another set of values for the predictor\n" + \
                                         "variables, or `n' to quit: ")
                    if new_data == "n": sys.exit(0)
                data_values = re.split( r'\s+', new_data )
                data_values = list(map(lambda x: int(x), data_values))
                if len(data_values) != self.X.shape[1]: 
                    print("Incorrect number of values entered. Aborting.")
                    sys.exit(1)
                self.Xtest = numpy.matrix(data_values)
                self.Ytest = (self.Xtest - self.mean0X) * self.B + self.mean0Y
                print("\nHere is the tuple of predictions for the data you entered:\n")
                print(self.Ytest)
                first_try = 0

    def PLS(self):
        """
        This implementation is based on the description of the algorithm by Herve
        Abdi in the article "Partial Least Squares Regression and Projection on
        Latent Structure Regression," Computational Statistics, 2010.  From my
        experiments with the different variants of PLS, this particular version
        generates the best regression results.  The Examples directory contains a
        script that carries out head-pose estimation using this version of PLS.
        """
        X,Y = self.X, self.Y
        self.mean0X = X.mean(0)
        if self.debug:
            print("\nColumn-wise mean for X:")
            print(self.mean0X)
        X = X - self.mean0X
        if self.debug:
            print("\nZero-mean version of X:")
            print(X)
        self.mean0Y = Y.mean(0)
        if self.debug:
            print("\nColumn-wise mean for Y is:")
            print(self.mean0Y)
        Y = Y - self.mean0Y
        if self.debug:
            print("\nZero-mean version of Y:")
            print(Y)
        T=U=W=C=P=Q=B=Bdiag=t=w=u=c=p=q=b=None        
        u = numpy.random.rand(1,self.N)
        u = numpy.asmatrix(u).T
        if self.debug:
            print("\nThe initial random guess for u: ")
            print(u)
        i = 0
        while (True):                             
            j = 0
            while (True):
                w = X.T * u
                w = w / numpy.linalg.norm(w)
                t = X * w
                t = t / numpy.linalg.norm(t)      
                c = Y.T * t
                c = c / numpy.linalg.norm(c)        
                u_old = u
                u = Y * c
                error = numpy.linalg.norm(u - u_old)
                if error < self.epsilon: 
                    if self.debug:
                        print("Number of iterations for the %dth latent vector: %d" % (i,j+1))
                    break
                j += 1    
            b = t.T * u
            b = b[0,0]
            if T is None:
                T = t
            else:
                T = numpy.hstack((T,t))
            if U is None:
                U = u
            else:
                U = numpy.hstack((U,u))
            if W is None:
                W = w
            else:
                W = numpy.hstack((W,w))
            if C is None:
                C = c
            else:
                C = numpy.hstack((C,c))
            p = X.T * t / (numpy.linalg.norm(t) ** 2)
            q = Y.T * u / (numpy.linalg.norm(u) ** 2)
            if P is None:
                P = p
            else:
                P = numpy.hstack((P,p))
            if Q is None:
                Q = q
            else:
                Q = numpy.hstack((Q,q))
            if Bdiag is None:
                Bdiag = [b]
            else:
                Bdiag.append(b)
            X_old = X
            Y_old = Y
            X = X - t * p.T
            Y = Y - b * t * c.T
            i += 1
            if numpy.linalg.norm(X) < 0.001: break
        if self.debug:
            print("\n\n\nThe T matrix:")
            print(T)
            print("\nThe U matrix:")
            print(U)
            print("\nThe W matrix:")
            print(W)
            print("\nThe C matrix:")
            print(C)
            print("\nThe P matrix:")
            print(P)
            print("\nThe b vector:")
            print(Bdiag)
            print("\nThe final deflated X matrix:")
            print(X)
            print("\nThe final deflated Y matrix:")
            print(Y)
        B = numpy.diag(Bdiag)
        B = numpy.asmatrix(B)  
        if self.debug:
            print("\nThe diagonal matrix B of b values:")
            print(B)
        self.B = numpy.linalg.pinv(P.T) * B * C.T
        if self.debug:
            print("\nThe matrix B of regression coefficients:")
            print(self.B)
        # For testing, make a prediction based on the original X:
        if self.debug:
            Y_predicted = (self.X - self.mean0X) * self.B
            print("\nY_predicted from the original X:")
            print(Y_predicted)
            Y_predicted_with_mean = Y_predicted + self.mean0Y
            print("\nThe predicted Y with the original Y's column-wise mean added:")
            print(Y_predicted_with_mean)
            print("\nThe original Y for comparison:") 
            print(self.Y)
        return self.B

    def PLS1(self):
        """
        This implementation is based on the description of the algorithm in the article
        "Overview and Recent Advances in Partial Least Squares" by Roman Rosipal and
        Nicole Kramer, LNCS, 2006.  Note that PLS1 assumes that the Y matrix consists
        of just one column. That makes it particularly appropriate for solving face
        recognition problems.  This module uses this method for a two-class
        discrimination between the faces.  We construct the X and Y matrices from the
        positive and the negative examples of the face to be recognized.  Each row of
        the X matrix consists of the vectorized representation of either a positive
        example of a face or a negative example.  The corresponding element in the
        one-column Y is +1 for the positive examples and -1 for the negative
        examples.
        """
        X,Y = self.X, self.Y
        if Y.shape[1] != 1:
            raise ValueError("PLS1 can only be called when the Y has only one column")
        self.mean0X = X.mean(0)
        X = X - self.mean0X
        self.mean0Y = Y.mean(0)
        Y = Y - self.mean0Y
        T=U=W=C=P=Q=B=t=w=u=c=p=q=None        
        u = Y
        i = 0
        while (True):
            w = X.T * u
            w = w / numpy.linalg.norm(w)
            t = X * w
            c = Y.T * t
            c = c / numpy.linalg.norm(c)        
            u = Y * c
            if T is None:
                T = t
            else:
                T = numpy.hstack((T,t))
            if U is None:
                U = u
            else:
                U = numpy.hstack((U,u))
            if W is None:
                W = w
            else:
                W = numpy.hstack((W,w))
            p = X.T * t / (numpy.linalg.norm(t) ** 2)
            q = Y.T * u / (numpy.linalg.norm(u) ** 2)
            if P is None:
                P = p
            else:
                P = numpy.hstack((P,p))
            if Q is None:
                Q = q
            else:
                Q = numpy.hstack((Q,q))
            X_old = X
            Y_old = Y
            X = X - t * p.T
            Y = Y - ( (t * t.T) * Y ) / (numpy.linalg.norm(t) ** 2)
            i += 1
            if numpy.linalg.norm(X) < 0.001: break
        if self.debug:
            print("\n\n\nThe T matrix:")
            print(T)
            print("\nThe U matrix:")
            print(U)
            print("\nThe W matrix:")
            print(W)
            print("\nThe C matrix:")
            print(C)
            print("\nThe X matrix:")
            print(X)
            print("\nThe Y matrix:")
            print(Y)
        self.B = W * ((P.T * W).I) * T.T * self.Y
        if self.debug:
            print("\nThe matrix B of regression coefficients:")
            print(self.B)
        return self.B

    def PLS2(self):
        """
        This implementation is based on the description of the algorithm in the article
        "Overview and Recent Advances in Partial Least Squares" by Roman Rosipal and
        Nicole Kramer, LNCS, 2006.  Unlike PLS1, this implementation places no
        constraints on the number of columns in the Y matrix.
        """
        X,Y = self.X, self.Y
        self.mean0X = X.mean(0)
        if self.debug:
            print("\ncolumn-wise mean for X:")
            print(self.mean0X)
        X = X - self.mean0X
        self.mean0Y = Y.mean(0)
        if self.debug:
            print("\ncolumn-wise mean for Y:")
            print(self.mean0Y)
        Y = Y - self.mean0Y
        T=U=W=C=P=Q=B=t=w=u=c=p=q=None        
        u = numpy.random.rand(1,self.N)
        u = numpy.asmatrix(u).T
        if self.debug:
            print("\nu vector initialization: ")
            print(u)
        i = 0
        while (True):
            j = 0
            while (True):
                w = X.T * u
                w = w / numpy.linalg.norm(w)
                t = X * w
                c = Y.T * t
                c = c / numpy.linalg.norm(c)        
                u_old = u
                u = Y * c
                error = numpy.linalg.norm(u - u_old)
                if error < self.epsilon: 
                    if self.debug:
                        print("Number of iterations for the %dth latent vector: %d" % (i,j+1))
                    break
                j += 1    
            if T is None:
                T = t
            else:
                T = numpy.hstack((T,t))
            if U is None:
                U = u
            else:
                U = numpy.hstack((U,u))
            if W is None:
                W = w
            else:
                W = numpy.hstack((W,w))
            if C is None:
                C = c
            else:
                C = numpy.hstack((C,c))
            p = X.T * t / (numpy.linalg.norm(t) ** 2)
            q = Y.T * u / (numpy.linalg.norm(u) ** 2)
            if P is None:
                P = p
            else:
                P = numpy.hstack((P,p))
            if Q is None:
                Q = q
            else:
                Q = numpy.hstack((Q,q))
            X_old = X
            Y_old = Y
            X = X - t * p.T
            Y = Y - ( (t * t.T) * Y ) / (numpy.linalg.norm(t) ** 2)
            i += 1
            if numpy.linalg.norm(X) < 0.001: break
        if self.debug:
            print("\n\n\nThe T matrix:")
            print(T)
            print("\nThe U matrix:")
            print(U)
            print("\nThe W matrix:")
            print(W)
            print("\nThe C matrix:")
            print(C)
        print("\nThe final deflated X matrix:")
        print(X)
        print("\nThe final deflated Y matrix:")
        print(Y)
        self.B = W * (P.T * W).I * C.T        
        if self.debug:
            print("\nThe matrix B of regression coefficients:")
            print(self.B)
            if self.Y.shape[1] > 1:
                Y_predicted = (self.X - self.mean0X) * self.B
                print("\nY_predicted from the original X:")
                print(Y_predicted)
                Y_predicted_with_mean = Y_predicted + self.mean0Y
                print("\nThe predicted Y with the original Y's column-wise mean added:")
                print(Y_predicted_with_mean)
                print("\nThe original Y for comparison:")
                print(self.Y)
        return self.B

    def vectorize_images_and_construct_X_and_Y_matrices_for_face_recognition_with_PLS1(self):
        """
        This method assumes that the images to be used for training and testing are
        organized as follows in the image_directory option supplied to the
        constructor of the module:

                                        image_directory
                                             |
                                             |
                           --------------------------------------
                          |                                      |
                          |                                      |
                       training                              testing
                          |                                      |
                          |                                      |
              -------------------------             -----------------------------
             |                         |           |                             |
             |                         |           |                             |
         positives                 negatives    positives                    negatives 


         The module constructs the X and the Y matrices from the images in the
         `training/positives' and the `training/negatives' subdirectories.  The
         vectorized representation of each image constitutes a row of the X
         matrix. The corresponding element in the one-column Y matrix is +1 for the
         images in the `positives' directory and -1 for the images in the `negatives'
         directory.  In a similar manner, the method constructs Xtest and Ytest
         matrices from the images in the `testing/positives' and `testing/negatives'
         subdirectories.
        """ 
        self._vectorize_images_for_PLS1()
        self._construct_X_and_Y_matrices_from_image_vectors_for_PLS1('training')
        self._construct_X_and_Y_matrices_from_image_vectors_for_PLS1('testing')
        if self.debug:
            print("\nThe X matrix:")
            print(self.X)
            print("\nThe Y matrix:")
            print(self.Y)
            print("\nThe Xtest matrix:")
            print(self.Xtest)
            print("\nThe Ytest matrix:")
            print(self.Ytest)

    def _vectorize_images_for_PLS1(self):
        os.chdir(self.image_directory)        
        try:
            assert 'training' and 'testing' in glob.glob('*')
        except:
            raise AssertionError('''The image directory must contain two subdirectories '''
                                 '''named `training' and `testing' ''')
        current_dir = os.getcwd()
        # Let's start with the images in training/positives/
        os.chdir(current_dir + "/training")        
        cwd = os.getcwd()
        try:
            assert 'positives' and 'negatives' in glob.glob('*')
        except:
            raise AssertionError('''The directory ''' + cwd + ''' must contain two '''
                                 '''subdirectories named `positives' and `negatives' ''')
        curr_dir = os.getcwd() 
        os.chdir(curr_dir + "/positives")        
        cwd = os.getcwd()
        for image in glob.glob('*.' + self.image_type):    
            if image.endswith( self.image_type ):    
                pixel_list = self.extract_pixels_from_image(image)
                self.training_positives.append(numpy.matrix(pixel_list))
        # Let's now process training/negatives/
        os.chdir(curr_dir + "/negatives")        
        cwd = os.getcwd()
        for image in glob.glob('*.' + self.image_type):    
            if image.endswith( self.image_type ):    
                pixel_list = self.extract_pixels_from_image(image)
                self.training_negatives.append(numpy.matrix(pixel_list))
        # Let's start with the images in testing/positives/
        os.chdir(current_dir + "/testing")        
        cwd = os.getcwd()
        try:
            assert 'positives' and 'negatives' in glob.glob('*')
        except:
            raise AssertionError('''The directory ''' + cwd + ''' must contain two '''
                                 '''subdirectories named `positives' and `negatives' ''')
        curr_dir = os.getcwd() 
        os.chdir(curr_dir + "/positives")        
        cwd = os.getcwd()
        for image in glob.glob('*.' + self.image_type):    
            if image.endswith( self.image_type ):    
                pixel_list = self.extract_pixels_from_image(image)
                self.testing_positives.append(numpy.matrix(pixel_list))
                self.testing_positives_filenames.append(image)
        # Let's now process testing/negatives/
        os.chdir(curr_dir + "/negatives")        
        cwd = os.getcwd()
        for image in glob.glob('*.' + self.image_type):    
            if image.endswith( self.image_type ):    
                pixel_list = self.extract_pixels_from_image(image)
                self.testing_negatives.append(numpy.matrix(pixel_list))
                self.testing_negatives_filenames.append(image)

    def vectorize_images_and_construct_X_and_Y_matrices_for_head_pose_estimation_with_PLS(self):
        """
        This method assumes that the image directory contains two subdirectories named:

            --  training

            --  testing

        Furthermore, the method assumes that the name of each image file in the two
        subdirectories named above is an encoding of the roll, pitch, and yaw values
        associated with the face image in that image.  For example, the name of the
        first image file in the directory `/head_pose_images/training/' is

                y1p1r2.jpg

        This name implies that the pose of the head in this image corresponds to the
        following values for roll, pitch, and yaw:

                yaw   = -30 degrees
                pitch = -30 degrees
                roll  = -20 degrees

        To understand why the name of the file translates into the values shown
        above, note that the pose of the head is varied with respect to each of the
        roll, pitch, and yaw parameters from -30 degrees to +30 degrees. We use the
        following mapping between the integer indices associated with the parameters
        in the file names and their actual angles:

               1   =>    -30 deg
               2   =>    -20 deg
               3   =>    -10 deg
               4   =>     0  deg
               5   =>    +10 deg
               6   =>    +20 deg
               7   =>    +30 deg

         This naming convention makes it easy to to create the rows of the Y matrix
         for each row of the X matrix.  Each row of the X matrix is the vectorized
         representation of the pixels in the image and each corresponding row of the
         Y matrix consists of the three pose angles associated with that image.
        """
        os.chdir(self.image_directory)        
        try:
            assert 'training' and 'testing' in glob.glob('*')
        except:
            raise AssertionError('''The image directory must contain two subdirectories '''
                                 '''names `training' and `testing' ''')
        current_dir = os.getcwd()
        os.chdir(current_dir + "/training")        
        pattern = r'y(.)p(.)r(.)'
        for imagename in glob.glob('*.' + self.image_type):    
            if imagename.endswith( self.image_type ):    
                pixel_list = self.extract_pixels_from_image(imagename)
                if self.X is None:
                    self.X = numpy.matrix(pixel_list)
                else:
                    self.X = numpy.vstack((self.X, pixel_list))
                basename = os.path.splitext(imagename)[0]                  
                m = re.search(pattern, basename)
                yaw,pitch,roll = m.group(1),m.group(2),m.group(3)
                yaw,pitch,roll = (int(yaw)-4)*10,(int(pitch)-4)*10,(int(roll)-4)*10
                if self.Y is None:
                    self.Y = numpy.matrix([yaw,pitch,roll])
                else:
                    self.Y = numpy.vstack((self.Y, [yaw,pitch,roll]))
        self.N = self.X.shape[0]
        if self.debug:
            print("\nThe X matrix:")
            print(self.X)
            print("\nThe Y matrix:")
            print(self.Y)
        os.chdir(current_dir + "/testing")                       
        for imagename in glob.glob('*.' + self.image_type):    
            if imagename.endswith( self.image_type ):    
                pixel_list = self.extract_pixels_from_image(imagename)
                if self.Xtest is None:
                    self.Xtest = numpy.matrix(pixel_list)
                else:
                    self.Xtest = numpy.vstack((self.Xtest, pixel_list))
                basename = os.path.splitext(imagename)[0]                  
                m = re.search(pattern, basename)
                yaw,pitch,roll = m.group(1),m.group(2),m.group(3)
                yaw,pitch,roll = (int(yaw)-4)*10,(int(pitch)-4)*10,(int(roll)-4)*10
                if self.Ytest is None:
                    self.Ytest = numpy.matrix([yaw,pitch,roll])
                else:
                    self.Ytest = numpy.vstack((self.Ytest, [yaw,pitch,roll]))
        if self.debug:
            print("\nThe Xtest matrix:")
            print(self.Xtest)
            print("\nThe Ytest matrix:")
            print(self.Ytest)    
 
    def _construct_X_and_Y_matrices_from_image_vectors_for_PLS1(self, training_or_testing):
        if training_or_testing is 'training':
            for vector in self.training_positives:
                if self.X is None:
                    self.X = numpy.matrix(vector)
                else:
                    self.X = numpy.vstack((self.X, vector))
                if self.Y is None:
                    self.Y = numpy.matrix([1])
                else:
                    self.Y = numpy.vstack((self.Y, numpy.matrix([1])))
            for vector in self.training_negatives:
                if self.X is None:
                    self.X = numpy.matrix(vector)
                else:
                    self.X = numpy.vstack((self.X, vector))
                if self.Y is None:
                    self.Y = numpy.matrix([-1])
                else:
                    self.Y = numpy.vstack((self.Y, numpy.matrix([-1])))
            self.N = self.X.shape[0]
        elif training_or_testing is 'testing':
            for vector in self.testing_positives:
                if self.Xtest is None:
                    self.Xtest = numpy.matrix(vector)
                else:
                    self.Xtest = numpy.vstack((self.Xtest, vector))
                if self.Ytest is None:
                    self.Ytest = numpy.matrix([1])
                else:
                    self.Ytest = numpy.vstack((self.Ytest, numpy.matrix([1])))
            for vector in self.testing_negatives:
                if self.Xtest is None:
                    self.Xtest = numpy.matrix(vector)
                else:
                    self.Xtest = numpy.vstack((self.Xtest, vector))
                if self.Ytest is None:
                    self.Ytest = numpy.matrix([-1])
                else:
                    self.Ytest = numpy.vstack((self.Ytest, numpy.matrix([-1])))
        if self.debug:
            print("\nThe size of the X matrix:")
            print(self.X.shape)  
            print("\nprinting out the Y matrix:")
            print(self.Y)

    def extract_pixels_from_image(self, imagename):
        if self.debug:
            cwd = os.getcwd()
            print("\nimage name is: ", cwd + "/" + imagename)
        im = Image.open(imagename)
        im = im.convert('L')        ## convert to gray level
        im.thumbnail(self.image_size_for_computations, Image.ANTIALIAS)
        width,height = im.size
        if self.debug: print("width: %d    height: %d" % (width, height))
        diff_width = self.image_size_for_computations[0] - width
        diff_height = self.image_size_for_computations[1] - height
        even_diff_width = even_diff_height = None
        if diff_width % 2 == 0: even_diff_width = True
        if diff_height % 2 == 0: even_diff_height = True
        pixel_list = []
        for i in range(height + diff_height):
            for j in range(width + diff_width):
                if i < diff_height // 2:
                    pixel_list.append(0.0)
                elif i >= height + diff_height // 2:
                    pixel_list.append(0.0)                    
                elif j < diff_width // 2 or j >= (diff_width // 2) + width:
                    pixel_list.append(0.0)                          
                else:
                    pixel_list.append(im.getpixel((j-(diff_width // 2),i-(diff_height // 2))))
        size_of_pixel_list = len(pixel_list)
        if size_of_pixel_list != self.image_size_for_computations[0] * \
                                 self.image_size_for_computations[1]:
            print("Image resizing step is not correct. Aborting.")
            sys.exit(1)
        return pixel_list

    def run_evaluation_of_PLS_regression_for_head_pose_estimation(self):
        """
        The docstring associated with the method

           vectorize_images_and_construct_X_and_Y_matrices_for_head_pose_estimation_with_PLS()

        applies here also. The method here uses the Xtest and Yest matrices
        constructed by the `vectorize' method named above from the images in the
        `testing' directory for evaluating PLS regression for head pose estimation.
        """
        if self.Xtest is None:
            raise ValueError("There is no data in your Xtest and Ytest matrices.  Aborting.")
        Y_predicted = (self.Xtest - self.mean0X) * self.B  + self. mean0Y
        error = numpy.linalg.norm(Y_predicted - self.Ytest) / (self.Ytest.shape[0] * 3)
        print("\nAverage error in head pose estimation: " + str(error) +  " degrees")
        print("\nThe error shown above was calculated by (1) taking the Frobenius norm of\n" +\
              "the difference between the true Y matrix (for just the data in the `testing'\n" +\
              "directory) and its value estimated by PLS regression; (2) Dividing the norm\n" +\
              "by the number of rows in Y to calculate the error per observation; and,\n" +\
              "finally, by (3) dividing the result by 3 to estimate the error per degree\n" +\
              "of freedom.")
        side_by_side_comparison = numpy.hstack((self.Ytest, Y_predicted))
        if sys.version_info[0] == 3:
            answer = input("\n\nWould you like to see a side-by-side comparison of the\n" +\
                           "the true values for the pose parameters and the values\n" +\
                           "as computed by PLS regression? Answer `y' for yes and\n" +\
                           "`n' for no: ")
        else:
            answer = raw_input("\n\nWould you like to see a side-by-side comparison of the\n" +\
                               "the true values for the pose parameters and the values\n" +\
                               "as computed by PLS regression? Answer `y' for yes and\n" +\
                               "`n' for no: ")
        if answer == 'y':
            print("\nDisplay of a side by side comparison.  The first three columns show the\n" + \
                  "true values for the head pose parameters and the last three columns show\n" +\
                  "the values as estimated by PLS regression.\n")
            print(side_by_side_comparison)
        return side_by_side_comparison


    def run_evaluation_of_PLS_regression_for_face_recognition(self):
        """
        The docstring associated with the method

        vectorize_images_and_construct_X_and_Y_matrices_for_face_recognition_with_PLS1()

        applies here also. The method here uses the Xtest and Yest matrices
        constructed by the `vectorize' method named above from the images in the
        `testing/positives' and the /testing/negatives/ subdirectories for evaluating
        PLS regression for face recognition.
        """
        if self.Xtest is None:
            raise ValueError("There is no data in your Xtest and Ytest matrices.  Aborting.")
        Y_predicted = (self.Xtest - self.mean0X) * self.B  + self.mean0Y
        if self.debug:
            print("\nPrinting the predicted Y:")
            print(Y_predicted)
        y_vals = Y_predicted[:,0].flatten().tolist()[0]
        if self.debug:
            print("predicted values as a list: ", y_vals)
        minval,maxval = min(y_vals),max(y_vals)
        if self.debug:
            print("min value: ", minval, "  and max value: ", maxval)
        delta = (maxval - minval) / 100.0
        hist = [0] * 101
        if self.debug:
            print("\nThe bin structure of the histogram --- with empty bins")
            print(hist)
        for val in y_vals:
            bin_index = int((val - minval) / delta)
            hist[bin_index] += 1
        if self.debug:
            print("\nThe populated histogram:")
            print(hist)                 
        total_count = functools.reduce(lambda x,y: x+y, hist)
        coarseness = 8   
        probs = [functools.reduce(lambda x,y: x+y, \
                             hist[coarseness*i:coarseness*i+coarseness])/float(total_count)
                                                 for i in range(int(len(hist)/coarseness))]
        prob_times_graylevel = [coarseness * i * probs[i] for i in range(len(probs))]
        mu_T = functools.reduce(lambda x,y: x+y, prob_times_graylevel)       # mean for the image
        prob_times_graysquared = [(coarseness * i - mu_T)**2 * probs[i] for i in range(len(probs))]
        sigma_squared_T = functools.reduce(lambda x,y: x+y, prob_times_graysquared)
        m0 = [functools.reduce(lambda x,y: x+y, probs[:k]) for k in range(1,len(probs)+1)]
        m1 = [functools.reduce(lambda x,y: x+y, prob_times_graylevel[:k]) for k in range(1,len(probs)+1)]
        sigmaB_squared = [None] * len(m0)     # for between-class variance as a func of threshold
        sigmaW_squared = [None] * len(m0)     # for within-class variance as a func of threshold
        variance_ratio = [None] * len(m0)     # for the ratio of the two variances
        for k in range(len(m0)):
            if 0 < m0[k] < 1.0:
                sigmaB_squared[k] = (mu_T * m0[k] - m1[k])**2 / (m0[k] * (1.0 - m0[k]))
                sigmaW_squared[k] = sigma_squared_T - sigmaB_squared[k]
                variance_ratio[k] = sigmaB_squared[k] / sigmaW_squared[k]
        otsu_threshold_testdata = variance_ratio.index(max(variance_ratio)) * coarseness
        otsu_threshold_testdata = otsu_threshold_testdata * delta + minval
        if self.debug: print("\nbest threshold for test data: ", otsu_threshold_testdata)
        self.decision_threshold_testdata = otsu_threshold_testdata
        if self.debug:
            print("\n\nThe testdata decision threshold for binary recognition: ", \
                                                          self.decision_threshold_testdata) 
        Y_predicted_as_list = Y_predicted[:,0].flatten().tolist()[0]
        Y_predicted_thresholded = \
            list(map(lambda x: 1 if x > self.decision_threshold_testdata else -1, Y_predicted_as_list))
        if self.debug: 
            print(Y_predicted_thresholded)
        Y_predicted_thresholded = numpy.matrix(Y_predicted_thresholded).T
        Y_comparison = numpy.hstack((self.Ytest, Y_predicted_thresholded))      
        if self.debug:
            print('''\nShowing computed and true labels side by side. Left column is '''  + \
                  '''computed labels and the right column the true labels:''')
            print(Y_comparison)  
        confusion_matrix = numpy.zeros(shape=(2,2))
        confusion_matrix = numpy.asmatrix(confusion_matrix)
        testing_pos_images_identified_as_neg = []
        testing_neg_images_identified_as_pos = []
        total_num_testing_pos = len(self.testing_positives_filenames)
        i = 0
        for row in Y_comparison:
            if (row[0,0] == 1) and (row[0,1] == 1):
                confusion_matrix[0,0] += 1            
            elif (row[0,0] == 1) and (row[0,1] == -1):
                confusion_matrix[1,0] += 1            
                testing_pos_images_identified_as_neg.append(self.testing_positives_filenames[i]) 
            elif (row[0,0] == -1) and (row[0,1] == -1):
                confusion_matrix[1,1] += 1                           
            elif (row[0,0] == -1) and (row[0,1] == 1):
                confusion_matrix[0,1] += 1   
                testing_neg_images_identified_as_pos.append(self.testing_negatives_filenames[i - \
                                                                          total_num_testing_pos])
            i += 1
        print("\nDisplaying the confusion matrix: \n\n")
        print("                  true pos     true neg   ")
        print("                 ------------------------" )
        print("                                         " )
        print("computed pos:     " + str(confusion_matrix[0,0]) + "         " + str(confusion_matrix[0,1])) 
        print("                                      ")
        print("computed neg:     " + str(confusion_matrix[1,0]) + "         " + str(confusion_matrix[1,1])) 
        print("\n")
        true_positive_detection_rate = confusion_matrix[0,0] / confusion_matrix[:,0].sum()
        false_positive_detection_rate = confusion_matrix[0,1] / confusion_matrix[:,1].sum()
        print("\nEstimated probability for true positives: " + str(true_positive_detection_rate))
        print("\nEstimated probability for false positives: " + str(false_positive_detection_rate)) 
        if sys.version_info[0] == 3:
            answer = input("\nWould you like to see the names of the image files in the\n" +\
                           "`testing' directory that were misclassified? Answer `y' for\n" +\
                           "yes and `n' for no: ")
        else:
            answer = raw_input("\nWould you like to see the names of the image files in the\n" +\
                               "`testing' directory that were misclassified? Answer `y' for\n" +\
                               "yes and `n' for no: ")
        if answer == 'n': 
            sys.exit(0)
        else:
            alist = sorted(testing_pos_images_identified_as_neg, key = lambda x: int(x.partition('.')[0]))
            blist = sorted(testing_neg_images_identified_as_pos, key = lambda x: int(x.partition('.')[0]))
            print("\npositive images in the `testing' directory misclassified as negatives: " + str(alist))
            print("\nnegative images in the `testing' directory misclassified as positives: " + str(blist))

        
#----------------------- End of PartialLeastSquares Class Definition  --------------------------

#----------------------------------    Test code follows      ----------------------------------

if __name__ == '__main__': 

    XMatrix_file = "X_data.csv"
    YMatrix_file = "Y_data.csv"

    pls = PartialLeastSquares( 
               XMatrix_file =  XMatrix_file,
               YMatrix_file =  YMatrix_file,
               epsilon      = 0.001,
               debug = 1,
            )
    pls.get_XMatrix_from_csv()
    pls.get_YMatrix_from_csv()
    pls.PLS()

