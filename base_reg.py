"""Base class for logistic and linear regressions."""

from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=C0103,R0913


class BaseReg(metaclass=ABCMeta):
    """Base regression class, which is thought to be super class for linear and
    logistic regression child classes. It implements the following public methods:
    gradient_descent, optimizer, alpha_selection_helper. It requires child
    classes have 'costs' property implemented. It's a dictionary with loss functions.
    """

    @property
    @abstractmethod
    def costs(self):
        """Attribute containing loss functions.
        It has to be implemented when this class is inherited."""


    def __init__(self, x, y, theta=None, add_bias=True, normalize=True):
        """Initialized by a child class, which passes the following arguments.

        :param x: 2D array; training data.
        :param y: 1D array; target data.
        :param theta: array of (1, n) dimensions. Optional. If None (default), 0s will be set.
        :param add_bias: bool; If True, 1s for intercept term will be added automatically.
        :param normalize: bool; Default True. Applies mean normalization and features scaling.
        """
        self.data, self.mean, self.std = \
            self.feature_normalize(x.astype(float)) if normalize else (x.astype(float), None, None)

        self.data = np.c_[np.ones((self.data.shape[0], 1)), self.data] if add_bias else self.data
        self.target = y.reshape(self.data.shape[0], 1)
        self.init_theta = theta if theta else np.zeros([1, self.data.shape[1]])


    @staticmethod
    def feature_normalize(x):
        """2D array as input where columns represent features and rows represent examples.
        Returns normalized data x_norm, together with mean and std values.
        """
        mean = sum(x) / x.shape[0]  # or np.mean(x, axis=0)
        std = np.power(sum(np.power(x - mean, 2)) / x.shape[0], 0.5)  # or np.std(x, axis=0)

        # Subtract the mean and scale the feature values by their standard deviations.
        x_norm = (x - mean) / std
        return x_norm, mean, std


    def gradient_descent(self, alpha, iterations, cost_fnt, Lambda=None, cost_history=False):
        """Performs gradient descent and returns evaluated theta weights.

        :param alpha: float; learning rate.
        :param iterations: int; number of iterations.
        :param cost_fnt: str; cost function e.g. 'mse'.
        :param Lambda: int or float; regularization hyperparameter.
        :param cost_history: bool; cost results collected with each iteration.
        :return: ndarray (1, n) dimension; thetas.
        """
        j_history = []

        theta = self.init_theta.copy()
        args = (theta, self.data, self.target, Lambda)

        for idx in range(iterations):
            if cost_history:
                j_history.append(self.costs[cost_fnt]['cost'](*args))
            else:
                # Compute only the last cost.
                if iterations - 1 == idx:
                    j_history.append(self.costs[cost_fnt]['cost'](*args))

            theta -= alpha * self.costs[cost_fnt]['gradient'](*args)

        return theta, j_history


    def optimizer(self, cost_fnt, iterations=None, Lambda=None, method='TNC'):
        """It's a wrapper of scipy.optimize.minimize function.

        :param cost_fnt: str; applicable costs functions names are in self.costs.keys().
        :param iterations: int; Max number of iterations. if None, then it's at least 100.
        :param Lambda: int o float; regularization hyperparameter.
        :param method: str; one of scipy.optimize.minimize methods. Default - 'TNC'.
        :return: ndarray of thetas and cost value.
        """
        from scipy import optimize

        res = optimize.minimize(
            fun=self.costs[cost_fnt]['cost'],
            x0=self.init_theta,
            args=(self.data, self.target, Lambda),
            method=method,
            jac=self.costs[cost_fnt]['gradient'],
            options={'maxiter': iterations}
        )

        weights = res.x
        weights = weights.reshape(1, len(weights))
        cost = res.fun
        return weights, cost


    def alpha_selection_helper(self, learning_rates_list, iterations, cost_fnt, Lambda=None):
        """Plot different learning rates to compare their performance."""

        for alpha in learning_rates_list:
            _, j_hist = self.gradient_descent(
                alpha, iterations, cost_fnt, Lambda=Lambda, cost_history=True)

            plt.plot(list(range(len(j_hist))), j_hist, '-')

        plt.legend(learning_rates_list)
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost')
        plt.show()
