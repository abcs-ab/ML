"""The module contains tools used by regression and neural network implementations.

MLFuncts class wraps the following methods:
mse_cost - mean squared error cost function.
mse_grad - gradient of mean squared error cost function.
sigmoid
sigmoid_derivative
sigmoid_cost
sigmoid_cost_grad
regularize - computes Ridge type regularization for both, cost and gradient.

MLTools class wraps the following methods:
accuracy - model metrics.
f1_score - model metrics.
roll - brings back ndarrays to their original shapes.
unroll - flattens ndarrays.
map_features - populates features with their polynomial representations.
answers_to_matrix - translates labels into matrix.
init_weights - initialize random weights.
"""

import numpy as np

# pylint: disable=C0103,R0914


class MLFuncts:
    """Implementations of costs, gradients and regularization."""

    # MSE ------------------------------------
    @staticmethod
    def mse_cost(theta, x, y, Lambda=None):
        """Mean squared error cost function with optional regularization.

        :param theta: ndarray; vector of weights.
        :param x: 2D ndarray; training data, examples in rows, features in cols.
        :param y: ndarray; (n, 1) shape.
        :param Lambda: int, float; If value is passed, regularization term will be added.
        :return: float; cost.
        """
        theta = theta.reshape(1, x.shape[1])
        cost = np.sum(np.power(x @ theta.T - y, 2)) / (2 * len(x))
        if Lambda:
            cost += MLFuncts.regularize(theta, Lambda, x.shape[0], 'cost')
        return cost

    @staticmethod
    def mse_grad(theta, x, y, Lambda=None):
        """Gradient of mean squared error cost function with optional regularization.

        :param theta: ndarray; vector of weights.
        :param x: 2D ndarray; training data, examples in rows, features in cols.
        :param y: ndarray; (n, 1) shape.
        :param Lambda: int, float; If value is passed, regularization term will be added.
        :return: ndarray; gradient
        """
        theta = theta.reshape(1, x.shape[1])
        grad = sum((x @ theta.T - y) * x) / len(x)
        if Lambda:
            grad += MLFuncts.regularize(theta, Lambda, x.shape[0], 'grad').flatten()
        return grad


    # Sigmoid --------------------------------
    @staticmethod
    def sigmoid(h):
        """Sigmoid activation function."""
        g = 1 / (1 + np.exp(-h))
        return g

    @staticmethod
    def sigmoid_derivative(h):
        """Returns sigmoid derivative."""
        sigmoid_result = MLFuncts.sigmoid(h)
        return sigmoid_result * (1 - sigmoid_result)

    @staticmethod
    def sigmoid_cost(theta, x, y, Lambda=None):
        """Returns sigmoid cost function result. Regularization is optional.

        :param theta: ndarray; vector of weights.
        :param x: 2D ndarray; training data, examples in rows, features in cols.
        :param y: ndarray; (n, 1) shape.
        :param Lambda: int, float; If value is passed, regularization term will be added.
        :return: float; cost.
        """
        theta = theta.reshape(1, x.shape[1])
        h = x @ theta.T
        a = MLFuncts.sigmoid(h)
        cost = -(np.sum([y * np.log(a) + (1 - y) * np.log(1 - a)])) / x.shape[0]
        if Lambda:
            cost += MLFuncts.regularize(theta, Lambda, x.shape[0], 'cost')
        return cost

    @staticmethod
    def sigmoid_cost_grad(theta, x, y, Lambda=None):
        """Returns gradient of sigmoid cost function. Regularization is optional.

        :param theta: (1, n) or (n,) ndarray; vector of weights.
        :param x: 2D ndarray; training data, examples in rows, features in cols.
        :param y: ndarray; (n, 1) shape.
        :param Lambda: float; If 0 or None, regularization term won't be added.
        :return: ndarray; gradient.
        """
        theta = theta.reshape(1, x.shape[1])
        grad = sum((MLFuncts.sigmoid(x @ theta.T) - y) * x) / x.shape[0]

        if Lambda:
            grad += MLFuncts.regularize(theta, Lambda, x.shape[0], 'grad').flatten()
        return grad

    @staticmethod
    def regularize(theta, Lambda, m, what='cost'):
        """Returns regularization term for cost and gradient depending on
        choice passed by the 'what' parameter. Ridge type.

        :param theta: ndarray; weights.
        :param Lambda: int or float; lambda hyperparameter.
        :param m: int; total number of training examples.
        :param what: str; default 'cost'. Any other value returns
        regularization part for a gradient.
        :return: int for cost, ndarray for gradient.
        """
        theta_bias_excluded = theta[:, 1:]
        if what == 'cost':
            return Lambda * np.sum([theta_bias_excluded ** 2]) / (2 * m)

        # Intercept term is not regularized, that's why it's not a part of equation.
        # It's however needed to preserve correct dimension for upcoming addition of
        # the regularization term, thus 0 for theta0 is prepended.
        return np.c_[0, Lambda * theta_bias_excluded / m]


class MLTools:
    """Collection of preprocessing, postprocessing and metrics methods."""

    @staticmethod
    def accuracy(pred, true, rounding=True):
        """Returns percentage of correct predictions.

        :param pred: ndarray, predictions.
        :param true: ndarray, true y values.
        :param rounding: If True, rounds values to 0 decimal parts.
        :return: float, accuracy
        """
        pred = pred.round() if rounding else pred
        return np.mean(np.double(pred.ravel() == true.ravel())) * 100

    @staticmethod
    def f1_score(pred, true, weighted=False, print_detailed=False):
        """Takes predicted and actual y arrays. Returns F1-score.

        :param pred: ndarray, predictions.
        :param true: ndarray, true y values.
        :param weighted: bool, If True, 'weighted' score is returned, 'macro' otherwise.
        :param print_detailed: If True, redirects labels precision, recall, F1-score to stdout.
        :return: float, F1-score metrics.
        """
        pred = pred.flatten()
        true = true.flatten()
        joined = np.c_[pred, true]

        labels, num_of_samples = np.unique(true, return_counts=True)
        precision, recall, f1score = [], [], []
        for label in labels:
            predicted = joined[joined[:, 0] == label]  # true positive + false positive
            predicted_true = len(predicted[predicted[:, 1] == label])  # true positives
            actual = len(joined[joined[:, 1] == label])  # true positive + false negative

            if predicted_true == 0:
                precision.append(0)
                recall.append(0)
                f1score.append(0)
                continue

            prec = predicted_true / len(predicted)
            rec = predicted_true / actual
            precision.append(prec)
            recall.append(rec)
            f1score.append((2 * prec * rec) / (prec + rec))

        if print_detailed:
            print("Precision per class: {}\n"
                  "Recall per class: {}\n"
                  "F1-score per class: {}".format(precision, recall, f1score))

        if weighted:
            return round(np.array(f1score) @ num_of_samples / sum(num_of_samples) * 100, 2)

        return round(sum(f1score) / len(f1score) * 100, 2)

    @staticmethod
    def map_features(x1, x2, exp):
        """Populates features with their consecutive exponents.
        E.g. It maps them into polynomial terms up to the 'exp' power.
        It results in more complex hypothesis, which may suit some data
        sets better, but it's prone to overfitting at the same time.

        :param x1: ndarray; feature 1.
        :param x2: ndarray; feature 2.
        :param exp: int; power.
        :return: 2D ndarray. x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc
        """

        # 1s for the intercept term are not added here.
        hypo = np.empty([len(x1), 0])
        for i in range(1, exp + 1):
            for j in range(i + 1):
                hypo = np.c_[hypo, (x1 ** (i-j)) * (x2 ** j)]

        return hypo

    @staticmethod
    def unroll(arrays):
        """Takes a collection of arrays.
        Returns a flattened and concatenated single array.

        >>> x = np.arange(6).reshape(3, 2)
        >>> MLTools.unroll([x, x])
        array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
        """
        return np.concatenate([a.flatten() for a in arrays])

    @staticmethod
    def roll(arr, shapes):
        """Brings back shapes to a previously flattened array of weights.
        Returns weights dictionary where weights for L0 are under the 0 key and so on.

        >>> arr = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
        >>> shapes = [(3, 2), (3, 2)]
        >>> MLTools.roll(arr, shapes)
        {0: array([[0, 1],
               [2, 3],
               [4, 5]]), 1: array([[0, 1],
               [2, 3],
               [4, 5]])}
        """
        thetas = {}
        start_idx = 0
        for idx, shape in enumerate(shapes):

            dim1, dim2 = shape
            end_idx = start_idx + (dim1 * dim2)
            thetas[idx] = arr[start_idx:end_idx].reshape(dim1, dim2)
            start_idx = end_idx

        return thetas

    @staticmethod
    def answers_to_matrix(target_vector):
        """Create identity matrix representing
        different classes and map it with a target vector y.

        >>> MLTools.answers_to_matrix(np.array([0, 1, 1, 0]))
        array([[1., 0.],
               [0., 1.],
               [0., 1.],
               [1., 0.]])
        """
        number_of_classes = len(np.unique(target_vector))
        eye = np.identity(number_of_classes)
        target_vector = target_vector.astype(int)

        return eye[target_vector, :]

    @staticmethod
    def init_weights(L_num, L1_num, epsilon=None):
        """Returns initial random weights for symmetry breaking.
        Values are kept within a range, between -epsilon and epsilon.
        L_num denotes a number of units in layer L,
        while L1_num denotes a number of units in layer L + 1
        """

        # The way to choose right epsilon is as follows:
        # (6**0.5) / ((L_in + L_out) ** 0.5)
        if epsilon is None:
            epsilon = (6 ** 0.5) / ((L_num + L1_num) ** 0.5)

        theta = np.random.rand(L1_num, L_num) * 2 * epsilon - epsilon

        return theta


if __name__ == "__main__":
    import doctest
    doctest.testmod()
