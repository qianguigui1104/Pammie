import graderUtil
import pandas as pd
from sklearn.datasets import load_boston
import numpy as np
from utils import feature_normalize
import os
import shutil

############################################################
# Problem 3.1: Implementing Linear Regression 
############################################################

class TestCase:

    def __init__(self, grader, X, y):
        self.grader = grader
        self.X = X
        self.y = y
        self.loss = None
        self.grad = None
        self.correct_theta = None
        self.gd_history = None
        self.prediction = None

    def test_loss(self, regressor):
        regressor.theta = np.zeros((self.X.shape[1],))
        submission_loss, submission_grad = regressor.loss(self.X, self.y)
        self.grader.requireIsEqual(self.loss[0], submission_loss)
        self.grader.requireIsEqual(self.grad[0], submission_grad)

        regressor.theta = np.ones((self.X.shape[1],))
        submission_loss, submission_grad = regressor.loss(self.X, self.y)
        self.grader.requireIsEqual(self.loss[1], submission_loss)
        self.grader.requireIsEqual(self.grad[1], submission_grad)

    def test_training(self, regressor):
        regressor.theta = np.zeros((self.X.shape[1],))
        regressor.train(self.X, self.y)
        self.grader.requireIsEqual(self.correct_theta, regressor.theta)
    
    def test_predicting(self, regressor):
        self.grader.requireIsEqual(self.prediction, regressor.predict(self.X)[:10])

    def test_normal_equation(self, regressor):
        regressor.theta = np.zeros((self.X.shape[1],))
        self.grader.requireIsEqual(self.correct_theta, regressor.normal_equation(self.X, self.y))
    

if __name__ == "__main__":

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")
    grader = graderUtil.Grader()
    submission = grader.load('linear_regressor')
    test_regressor = submission.LinearReg_SquaredLoss()
    graderUtil.setTolerance(1e-6)
    
    # Load the housing test dataset.
    bdata = load_boston()
    df = pd.DataFrame(data=bdata.data, columns=bdata.feature_names)
    X = df.LSTAT
    housing_test_case = TestCase(grader, np.vstack([np.ones((X.shape[0],)), X]).T, bdata.target)

    ############################################################
    # -- 3.1.a: Linear Regression with One Variable


    def test_3_1_a1():
        """
        Testing the loss function.
        """
        loss = np.array([296.07345850, 155.43347855])
        grad = np.array([[-22.53280632, -236.75723123], [-8.87974308, -13.11017925]])
        housing_test_case.loss = loss
        housing_test_case.grad = grad
        housing_test_case.test_loss(test_regressor)

    
    grader.addPart('3.1.A1', test_3_1_a1, 5)

    
    def test_3_1_a2():
        """
        Testing the gradient descent.
        """
        housing_test_case.correct_theta = np.array([0.88635164, 1.07125317])
        housing_test_case.test_training(test_regressor)


    grader.addPart('3.1.A2', test_3_1_a2, 5)


    def test_3_1_a3():
        """
        Testing the prediction. Only test top 10 results.
        """
        prediction_result = [6.22119240, 10.67760557, 5.20350189, 4.03583594, 6.59613101, 6.46758063, 14.20202848, 21.40084975, 32.94895887, 19.20478076]
        housing_test_case.prediction = np.array(prediction_result)
        housing_test_case.test_predicting(test_regressor)

    grader.addPart('3.1.A3', test_3_1_a3, 5)

    ############################################################
    # -- 3.1.b: Linear Regression with Multiple Variables

    X = df.values
    y = bdata.target
    mult_submission = grader.load("linear_regressor_multi")
    

    def test_3_1_b1():
        """
        Testing the normlaize function.
        """
        file = open("grader_data/norm_test.txt", "r")
        x_str = file.read()
        correct_mu = np.array([3.61352356e+00, 1.13636364e+01, 1.11367787e+01, 6.91699605e-02,
        5.54695059e-01, 6.28463439e+00, 6.85749012e+01, 3.79504269e+00,
        9.54940711e+00, 4.08237154e+02, 1.84555336e+01, 3.56674032e+02,
        1.26530632e+01])
        correct_sigma = np.array([8.59304135e+00, 2.32993957e+01, 6.85357058e+00, 2.53742935e-01,
        1.15763115e-01, 7.01922514e-01, 2.81210326e+01, 2.10362836e+00,
        8.69865112e+00, 1.68370495e+02, 2.16280519e+00, 9.12046075e+01,
        7.13400164e+00])
        correct_x = np.array([[float(x_val) for x_val in x_row.split(",")] for x_row in x_str.split("\n")])
        X_norm, mu, sigma = feature_normalize(df.values)
        grader.requireIsEqual(correct_mu, mu)
        grader.requireIsEqual(correct_sigma, sigma)
        grader.requireIsEqual(correct_x, X_norm[:10, :])
    
    grader.addPart('3.1.B1', test_3_1_b1, 5)

    X_norm, mu, sigma = feature_normalize(X)
    multi_test_case = TestCase(grader, np.vstack([np.ones((X.shape[0],)),X_norm.T]).T, bdata.target)
    mult_regressor = mult_submission.LinearReg_SquaredLoss()


    def test_3_1_b2():
        """
        Test the mutlivariable loss and gradient descent.
        """
        loss = np.array([296.07345849, 301.90740686])
        grad1 = np.array([-22.53280632, 3.56774723, -3.31177597, 4.44447236,
        -1.61029253, 3.92622819, -6.38897522, 3.4634629, -2.29634809, 
        3.50638621, 4.30491357, 4.66554993, -3.06384186, 6.77765364])
        grad2 = np.array([-21.53280632, 6.46153158, -4.45417629, 7.63485292,
        -0.64575753, 7.02472697, -7.2769985, 6.27407833, -4.59765138, 
        7.18996587, 8.07955422, 6.81028022, -4.24526887, 9.49319148])
        multi_test_case.loss = loss
        multi_test_case.grad = [np.array(grad1), np.array(grad2)]
        multi_test_case.test_loss(mult_regressor)

        multi_test_case.correct_theta = np.array([2.14530011, -0.26831927, 0.24362291, -0.32576772, 0.14969465, -0.28034553,
        0.55075363, -0.243389, 0.12828532, -0.24210741, -0.31400086, -0.38293446, 0.23521995, -0.55768724])
        multi_test_case.test_training(mult_regressor)

    
    grader.addPart('3.1.B2', test_3_1_b2, 10)

    
    def test_3_1_b3():
        """
        Test the multivariable prediction.
        """
        prediction_result = [4.73108107, 3.60217514, 4.74466557, 4.9960257,  4.85810931, 4.26001269,
        3.36910153, 2.73937515, 1.41788231, 2.86814308]
        multi_test_case.prediction = np.array(prediction_result)
        multi_test_case.test_predicting(mult_regressor)


    grader.addPart('3.1.B3', test_3_1_b3, 5)


    def test_3_1_b4():
        """
        Test the normal equation.
        """
        multi_test_case.correct_theta = np.array([2.25328063e+01, -9.28146064e-01, 1.08156863e+00, 1.40899997e-01,
        6.81739725e-01, -2.05671827e+00, 2.67423017e+00, 1.94660717e-02, -3.10404426e+00, 2.66221764e+00, 
        -2.07678168e+00, -2.06060666e+00, 8.49268418e-01, -3.74362713e+00,])
        multi_test_case.test_normal_equation(mult_regressor)

    
    grader.addPart('3.1.B4', test_3_1_b4, 5)
    grader.grade()
