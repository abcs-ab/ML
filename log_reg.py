"""Logistic regression basic implementation."""

import numpy as np

from base_reg import BaseReg
from ml_toolbox import MLFuncts, MLTools as MLT

# pylint: disable=C0103,R0913


class LogReg(BaseReg):
    """Logistic regression implementation. The class inherits
    gradient descent, optimizer and alfa_selection_helper from BaseReg class.
    Instance public methods, the class implements itself are:
    one_vs_all, predict and get_metrics.
    """

    costs = {
        'sigmoid': {
            'cost': MLFuncts.sigmoid_cost,
            'gradient': MLFuncts.sigmoid_cost_grad},
    }

    def __init__(self, x, y, theta=None, add_bias=True, normalize=True):
        """Passes all arguments to BaseReg class, where all data is instantiated.

        :param x: 2D array; training data. (m, n) shape.
        :param y: 1D array; target data. (m,) shape.
        :param theta: array of (1, n) dimensions. Optional. If None (default), 0s will be set.
        :param add_bias: bool; If True, 1s for intercept term will be added automatically.
        :param normalize: bool; Default True. Applies mean normalization and features scaling.
        """
        super().__init__(x, y, theta=theta, add_bias=add_bias, normalize=normalize)
        self.normalize = normalize

    def one_vs_all(self, cost_fnt, iterations, Lambda=None, method='TNC'):
        """One vs all classifier. scipy.optimize.minimize under the hood.

        :param cost_fnt: str; applicable costs functions names are in self.costs.keys().
        :param iterations: int; Max number of iterations. if None, then it's at least 100.
        :param Lambda: int; regularization hyperparameter.
        :param method: str; one of scipy.optimize.minimize methods. Default - 'TNC'.
        :return: ndarray of thetas and labels.
        """

        # Get distinct labels sorted. It maps indices with labels, that
        # might be used later to reconstruct labels from predicted indices.
        labels = np.unique(self.target)

        all_thetas = np.empty([0, self.data.shape[1]])
        for label in labels:
            # target==label bool values are cast to 0 1 integers.
            temp = LogReg(self.data, (self.target == label).astype(int),
                          add_bias=False, normalize=False)

            thetas, cost = temp.optimizer(cost_fnt, iterations=iterations,
                                          Lambda=Lambda, method=method)

            print("Label {}, Cost: {}".format(label, cost))
            all_thetas = np.r_[all_thetas, thetas]

        # all_thetas[0, n] refers to labels[0].
        return all_thetas, labels


    def predict(self, x, theta):
        """Takes 2D ndarray of x and single theta vector of (1, n) shape.
        Input will be normalized if training data were normalized.
        """
        if self.normalize and x is not self.data:
            x = x.astype(float)
            x[:, 1:] = (x[:, 1:] - self.mean) / self.std

        h = x @ theta.T
        return MLFuncts.sigmoid(h)

    def get_metrics(self, theta, labels=None, weighted=True, details=False):
        """Returns f1-score and accuracy metrics.

        :param theta: ndarray; weights.
        :param labels: ndarray; Labels returned by one vs all method.
        :param weighted: bool; If True, 'weighted' score is returned, 'macro' otherwise.
        :param details: bool; prints metrics per class.
        :return: float; f1-score, accuracy
        """
        predictions = self.predict(self.data, theta)
        if predictions.shape[1] > 1:
            max_pred = np.argmax(predictions, axis=1)  # For multiple classes.
        else:
            max_pred = predictions.round()  # For single class.

        # Labels from one vs all method if they are somehow different from consecutive numbers
        # from 0 to k, which then could be used as indices directly.
        if labels is not None:
            max_pred = np.array(list(map(lambda i: labels[i], max_pred)))

        f1score = MLT.f1_score(max_pred, self.target, weighted=weighted, print_detailed=details)
        accuracy = MLT.accuracy(max_pred, self.target)

        return f1score, accuracy


if __name__ == "__main__":
    X = np.array([[0.51, 0.26, 0.71],
                  [0.3, 0.14, 0.18],
                  [0.2, 0.99, 0.18],
                  [0.11, 0.22, 0.44],
                  [0.48, 0.77, 0.61]])

    Y = np.array([[0], [0], [0], [1], [1]])

    print('Scipy optimizer -------------------------------')
    net = LogReg(X, Y, normalize=False)
    th, cost1 = net.optimizer('sigmoid', iterations=30, Lambda=0, method='TNC')

    f1s, acc = net.get_metrics(th, weighted=False)
    print("F1-score: {} Accuracy: {}\n".format(f1s, acc))


    print('Gradient descent ------------------------------')
    net2 = LogReg(X, Y, normalize=True)
    th2, cost2 = net2.gradient_descent(alpha=0.001, iterations=10, cost_fnt='sigmoid',
                                       Lambda=1, cost_history=False)

    f1s, acc = net2.get_metrics(th2, weighted=False)
    print("F1-score: {} Accuracy: {}\n".format(f1s, acc))


    print('One vs All ------------------------------------')
    net3 = LogReg(X, Y, normalize=False)
    all_theta, labels_list = net3.one_vs_all('sigmoid', iterations=10, Lambda=3)

    f1s, acc = net3.get_metrics(all_theta, labels=None, weighted=False, details=False)
    print("F1-score: {} Accuracy: {}".format(f1s, acc))
