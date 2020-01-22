import graderUtil
import pandas as pd
import numpy as np
import utils
from sklearn.preprocessing import PolynomialFeatures

############################################################
# Problem 3.2: Implementing Regularized Linear Regression 
############################################################

if __name__ == "__main__":

    grader = graderUtil.Grader()
    reg_submission = grader.load('reg_linear_regressor_multi')
    util_submission = grader.load('utils')
    test_regressor = reg_submission.RegularizedLinearReg_SquaredLoss()
    
    # Load the housing test dataset.
    X, y, Xtest, ytest, Xval, yval = utils.load_mat('ex2data1.mat')
    XX = np.vstack([np.ones((X.shape[0],)),X]).T

    poly = PolynomialFeatures(degree=6,include_bias=False)
    X_poly = poly.fit_transform(np.reshape(X,(len(X),1)))
    X_poly, mu, sigma = utils.feature_normalize(X_poly)

    # add a column of ones to X_poly

    XX_poly = np.vstack([np.ones((X_poly.shape[0],)),X_poly.T]).T
    print(X, XX_poly)

    # map Xtest and Xval into the same polynomial features

    X_poly_test = poly.fit_transform(np.reshape(Xtest,(len(Xtest),1)))
    X_poly_val = poly.fit_transform(np.reshape(Xval,(len(Xval),1)))

    # normalize these two sets with the same mu and sigma

    X_poly_test = (X_poly_test - mu) / sigma
    X_poly_val = (X_poly_val - mu) / sigma

    # add a column of ones to both X_poly_test and X_poly_val
    XX_poly_test = np.vstack([np.ones((X_poly_test.shape[0],)),X_poly_test.T]).T
    XX_poly_val = np.vstack([np.ones((X_poly_val.shape[0],)),X_poly_val.T]).T



    def test_3_2_a1():
        """
        Testing the ridge loss function.
        """
        grader.requireIsEqual(140.95412088, test_regressor.loss(np.zeros((XX.shape[1],)), XX, y, 0.1))
        grader.requireIsEqual(303.95569222, test_regressor.loss(np.ones((XX.shape[1],)), XX, y, 0.1))
        grader.requireIsEqual(303.99319222, test_regressor.loss(np.ones((XX.shape[1],)), XX, y, 1))

    
    grader.addPart('3.2.A1', test_3_2_a1, 5)

    
    def test_3_2_a2():
        """
        Testing the ridge gradient function.
        """
        grader.requireIsEqual(np.array([-11.21758933, -245.65199649]), test_regressor.grad_loss(np.zeros((XX.shape[1],)), XX, y, 0.1).ravel())
        grader.requireIsEqual(np.array([-15.30301567, 598.17574417]), test_regressor.grad_loss(np.ones((XX.shape[1],)), XX, y, 0.1).ravel())
        grader.requireIsEqual(np.array([-15.30301567, 598.25074417]), test_regressor.grad_loss(np.ones((XX.shape[1],)), XX, y, 1).ravel())
    

    grader.addPart('3.2.A2', test_3_2_a2, 5)
    lasso_regressor = reg_submission.LassoLinearReg_SquaredLoss()

    def test_3_2_a3():
        """
        Testing the lasso loss function.
        """
        grader.requireIsEqual(140.95412088, lasso_regressor.loss(np.zeros((XX_poly.shape[1],)), XX_poly, y, 0.1))
        grader.requireIsEqual(122.83669687, lasso_regressor.loss(np.ones((XX_poly.shape[1],)) / 2., XX_poly, y, 0.1))
        grader.requireIsEqual(123.06169687, lasso_regressor.loss(np.ones((XX_poly.shape[1],)) / 2., XX_poly, y, 1))

    
    grader.addPart('3.2.A3', test_3_2_a3, 5)

    
    def test_3_2_a4():
        """
        Testing the lasso gradient function.
        """
        correct_grad1 = np.array([-11.21758933, -10.5511193, -1.85859785, -8.73111503, 0.03463017, -6.95600977, 1.29621091])
        correct_grad2 = np.array([-10.71758933, -9.87438828, -1.21388082, -8.25553615, 0.53693007, -6.68459566, 1.67295436])
        correct_grad3 = np.array([-10.71758933, -9.79938828, -1.13888082, -8.18053615, 0.61193007, -6.60959566, 1.74795436])
        grader.requireIsEqual(correct_grad1, lasso_regressor.grad_loss(np.zeros((XX_poly.shape[1],)), XX_poly, y, 0.1).ravel())
        grader.requireIsEqual(correct_grad2, lasso_regressor.grad_loss(np.ones((XX_poly.shape[1],)) / 2., XX_poly, y, 0.1).ravel())
        grader.requireIsEqual(correct_grad3, lasso_regressor.grad_loss(np.ones((XX_poly.shape[1],)) / 2., XX_poly, y, 1).ravel())
    

    grader.addPart('3.2.A4', test_3_2_a4, 5)

    XXval = np.vstack([np.ones((Xval.shape[0],)),Xval]).T


    def test_3_2_a5():
        """
        Testing the learning curve function.
        """
        correct_et = np.array([7.24761496e-21, 1.50991955e-07, 3.28659525e+00, 2.84267780e+00,
        1.31540488e+01, 1.94439625e+01, 2.00985217e+01, 1.81728587e+01,
        2.26094054e+01, 2.32614616e+01, 2.43172496e+01, 2.23739065e+01])
        correct_ev = np.array([138.84677698, 110.33541087,  45.00640467,  48.36591385,
        35.86450109,  33.82923053,  31.97044225,  30.86206309,
        31.13568211,  28.93610404,  29.55139299,  29.43375392])
        error_train, error_val = util_submission.learning_curve(XX, y, XXval, yval, 0.1)
        grader.requireIsEqual(correct_et, error_train)
        grader.requireIsEqual(correct_ev, error_val)
        
        correct_et = np.array([1.09823899e-16, 3.74111029e-06, 3.28660012e+00, 2.84268040e+00,
        1.31540496e+01, 1.94439632e+01, 2.00985222e+01, 1.81728591e+01,
        2.26094057e+01, 2.32614618e+01, 2.43172498e+01, 2.23739066e+01])
        correct_ev = np.array([138.84677713, 110.47487912,  44.99110552,  48.35392784,
        35.86184714,  33.82630702,  31.96826804,  30.86053003,
        31.13441899,  28.93569052,  29.55123972,  29.43349724])
        error_train, error_val = util_submission.learning_curve(XX, y, XXval, yval, 0.5)
        grader.requireIsEqual(correct_et, error_train)
        grader.requireIsEqual(correct_ev, error_val)


    grader.addPart('3.2.A5', test_3_2_a5, 5)

    
    def test_3_2_a7():
        """
        Testing the validation curve function.
        """
        correct_et = np.array([ 0.19805294,  0.19904131,  0.20340439,  0.22402724,  0.28286279,
        0.44961435,  0.87865468,  1.91136448,  4.49754745, 15.24085796])
        correct_ev = np.array([22.11960785, 19.79551915, 17.29222383, 13.76732928, 10.42783308,
        7.01247601,  4.79684782,  3.93351001,  3.5409028 , 10.59606879])
        _, error_train, error_val = util_submission.validation_curve(XX_poly, y, XX_poly_val, yval)
        grader.requireIsEqual(correct_et, error_train)
        grader.requireIsEqual(correct_ev, error_val)

    
    grader.addPart('3.2.A7', test_3_2_a7, 5)
    grader.grade()
