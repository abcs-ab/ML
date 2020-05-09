"""Linear regression basic implementation.
Only "mean squared error" and normal equation at the moment,
but can easily be extended with other loss functions.
"""

import numpy as np

from base_reg import BaseReg
from ml_toolbox import MLFuncts

# pylint: disable=C0103,R0913


class LinReg(BaseReg):
    """Linear regression class. Inherits mostly from BaseReg while itself
    implements predict and normal_equation public methods.
    The "costs" class attribute has to be set with at least one cost function,
    which will later be used by inherited gradient descent method.
    """

    # Dictionary of loss functions. Mean squared error for example.
    # Other functions can easily be added. Everything what's needed is:
    # name, cost function and derivative.
    costs = {
        'mse': {
            'cost': MLFuncts.mse_cost,
            'gradient': MLFuncts.mse_grad}
    }

    def __init__(self, x, y, theta=None, add_bias=True, normalize=False):
        """Passes all arguments to BaseReg class, where all data is instantiated.

        :param x: 2D array; training data. (m, n) shape.
        :param y: 1D array; target data. (m,) shape.
        :param theta: array of (1, n) dimensions. Optional. If None (default), 0s will be set.
        :param add_bias: bool; If True, 1s for intercept term will be added automatically.
        :param normalize: bool; Default True. Applies mean normalization and features scaling.
        """
        super().__init__(x, y, theta=theta, add_bias=add_bias, normalize=normalize)
        self.normalize = normalize


    def normal_equation(self):
        """Returns thetas in ndarray of (1, n) dimensions."""
        theta = np.linalg.inv(self.data.T @ self.data) @ self.data.T @ self.target
        return theta.T


    def predict(self, x, theta):
        """Takes x 2D ndarray and theta ndarray of (1, n) dimensions.
        Predicts value based on previously learnt weights.
        """
        if self.normalize and x is not self.data:
            x = x.astype(float)
            x[:, 1:] = (x[:, 1:] - self.mean) / self.std

        return x @ theta.T
