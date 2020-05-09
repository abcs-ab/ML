"""Python implementation of fully connected neural network."""

import numpy as np
from scipy import optimize

from ml_toolbox import MLFuncts, MLTools as MLT

# pylint: disable=C0103,R0914,R0902


class Net:
    """Fully connected neural network with sigmoid activation.
    Scipy optimizer under the hood. Instance public methods:
    train, predict and get_metrics."""

    def __init__(self, x, hidden, y, theta=None):
        """Initialize all required attributes.

        :param x: 2D ndarray; examples in rows, features in cols.
        :param hidden: list of ints; heights of hidden layers, e.g. [20, 20, 15]
        :param y: 1D ndarray; vector of target classes consisting
        of consecutive numbers from 0 to k. e.g [1, 0, 1, 0, 2]. It's important, that
        labels are in form of consecutive integers, since they will be used as indices.
        :param theta: dict of ndarrays or None (default);
        If None, random weights will be initialized. Otherwise,
        correct ndarrays with bias term included have to be provided.
        """
        self.m = x.shape[0]  # number of examples.
        self.net = {0 : np.c_[np.ones([x.shape[0], 1]), x]}  # L0 with bias initialized.
        self.target = y.flatten()
        self.y2d = MLT.answers_to_matrix(self.target)  # y 2D ndarray.

        self.net_struct = self.get_structure(x.shape[1], hidden, self.y2d.shape[1])
        self.num_of_layers = len(self.net_struct)

        self.theta = theta
        if self.theta is None:
            self._initialize_theta()

        self.theta_shapes = [self.theta[k].shape for k in sorted(self.theta.keys())]

        self._initialize_net()
        self.delta = {}


    @staticmethod
    def get_structure(input_height, hidden_height, output_height):
        """Returns total number of units per layer, including bias term."""

        # Increase numbers of units in layers by 1 to include bias term.
        # It doesn't apply to output layer.
        hidden_height = [u + 1 for u in hidden_height]
        return [input_height + 1] + hidden_height + [output_height]


    def _initialize_theta(self):
        """Initialize theta with random values for all layers."""
        self.theta = {}
        for layer_idx in range(self.num_of_layers - 1):
            l1 = self.net_struct[layer_idx]
            l2 = self.net_struct[layer_idx + 1]

            # layers heights have bias included, thus 1 is subtracted from
            # the l2 in order to prevent it from creating an extra unit.
            # It doesn't apply to the number of output layer units.
            if layer_idx + 1 != self.num_of_layers - 1:
                l2 -= 1

            self.theta[layer_idx] = MLT.init_weights(l1, l2)


    def _initialize_net(self):
        """Sets zeros ndarrays for all hidden layers
        with ones for bias term applied.
        """
        for layer_idx in range(1, self.num_of_layers):
            total_units = self.net_struct[layer_idx]
            self.net[layer_idx] = np.zeros([self.m, total_units])

            if layer_idx < self.num_of_layers - 1:
                self.net[layer_idx][:, 0] = 1  # Set 1s for bias term.


    def _gradient_checking(self, grad, Lambda=None, epsilon=1e-4):
        """Computes numerical gradient and compares it with the one
        evaluated by back propagation. Returns ratio of distances
        between "two sides" neighbours of numerical gradient.

        :param grad: 1D ndarray; evaluated backprop gradient vector.
        :param Lambda: float; regularization hyperparameter.
        :param epsilon: float; the smaller epsilon, the better theta approx,
         but computations can be more expensive. The value should be small anyway.
        :return: float;
        """
        theta_vec = MLT.unroll([self.theta[k] for k in sorted(self.theta.keys())])
        grad_approx = []
        for idx in range(len(theta_vec)):
            temp = theta_vec[idx]
            theta_vec[idx] += epsilon
            cost_plus = self._cost_fnt(theta_vec, Lambda=Lambda)
            theta_vec[idx] = temp
            theta_vec[idx] -= epsilon
            cost_minus = self._cost_fnt(theta_vec, Lambda=Lambda)
            theta_vec[idx] = temp
            num_grad = (cost_plus - cost_minus) / (2 * epsilon)
            grad_approx.append(num_grad)

        # returns relative difference, which should be very small
        # if gradient's been evaluated correctly.
        dist_minus = np.linalg.norm(np.array(grad_approx) - grad)
        dist_plus = np.linalg.norm(np.array(grad_approx) + grad)
        print("Relative difference:", dist_minus / dist_plus)


    def sig_grad(self, layer_idx):
        """Takes layer index. Sigmoid function is already evaluated
        as units activator, so we can refer to it by the index.
        Returns sigmoid derivative.
        """
        return self.net[layer_idx] * (1 - self.net[layer_idx])


    def _forward_prop(self):
        """Activates network units in instance net dictionary,
        where keys correspond to consecutive network layers.
        Theta dictionary contains weights for each layer.
        Same keys in both dictionaries refers to the same layer.
        """
        l_out_idx = self.num_of_layers - 1
        for l in range(l_out_idx):
            hypo = self.net[l] @ self.theta[l].transpose()

            if l + 1 < l_out_idx:
                self.net[l + 1][:, 1:] = MLFuncts.sigmoid(hypo)
            else:
                # Output layer doesn't have a bias unit, thus no slicing needed.
                self.net[l + 1] = MLFuncts.sigmoid(hypo)


    def _cost_fnt(self, weights=None, Lambda=None):
        """Runs forward propagation and returns its cost.

        :param weights: 1D ndaaray, vector of weights passed by a scipy optimizer.
        If present, after being reshaped, it will update instance theta attribute.
        :param Lambda: int, float; If 0 or None, regularization term won't be added.
        :return: float;
        """

        if weights is not None:
            self.theta = MLT.roll(weights, self.theta_shapes)

        self._forward_prop()

        l_out_idx = self.num_of_layers - 1
        cost = -(np.sum([self.y2d * np.log(self.net[l_out_idx]) +
                         (1 - self.y2d) * np.log(1 - self.net[l_out_idx])])) / self.m

        # Cost regularization. Bias terms are not regularized.
        if Lambda is not None:
            sigma = sum([np.sum([t[:, 1:] ** 2]) for t in self.theta.values()])
            cost += Lambda * sigma / (2 * self.m)

        return cost


    def _back_prop(self, _weights=None, Lambda=None, _gradient_check=False):
        """Back propagation. Returns gradient vector.

        :param _weights: Not used, but needed as a placeholder. The class uses
        instance theta attribute for required computations instead of the one
        passed by the optimizer. Theta is set when cost_fnt is called.
        :param Lambda: int, float; If 0 or None, regularization term won't be added.
        :param _gradient_check: bool, If True, turns on gradient checking, used only
        to check whether or not back_prop computes the right gradient. It's a one time
        validator and should be set to False when implementation is fine.
        :return: 1D ndarray, gradient vector
        """
        del _weights

        gradient = []
        for l in range(1, self.num_of_layers).__reversed__():
            if l == self.num_of_layers - 1:
                self.delta[l] = self.net[l] - self.y2d
            else:
                self.delta[l] = self.delta[l + 1] @ self.theta[l][:, 1:] \
                                * self.sig_grad(l)[:, 1:]

            gradient.append((self.delta[l].transpose() @ self.net[l-1]) / self.m)

            # Gradient regularization. Bias terms are not regularized.
            if Lambda is not None:
                gradient[-1][:, 1:] += Lambda * self.theta[l - 1][:, 1:] / self.m

        # Reverse list of gradient 2D arrays, to preserve correct order.
        # Flatten and concat the arrays into one vector.
        grad_vec = MLT.unroll(gradient[::-1])

        if _gradient_check:
            self._gradient_checking(grad_vec, Lambda=Lambda)

        return grad_vec


    def train(self, iterations=1, Lambda=None, method='L-BFGS-B'):
        """Wraps a scipy.optimize.minimize function. Returns cost only,
        since weights are updated with every iteration at the instance level.
        They are held by theta attribute. Not all scipy optimizing methods will work,
        since some of them need additional arguments like hessian for example.

        :param iterations: int; number of iterations.
        :param Lambda: int, float; If 0 or None, regularization term won't be added.
        :param method: str; Recommended methods: L-BFGS-B or TNC.
        :return: float; cost value.
        """
        x0 = MLT.unroll(self.theta[k] for k in sorted(self.theta.keys()))

        res = optimize.minimize(
            fun=self._cost_fnt,
            x0=x0,
            args=Lambda,
            method=method,
            jac=self._back_prop,
            options={'maxiter': iterations}
        )

        return res.fun

    def predict(self, test_data):
        """Predicts class for a given data. Forward prop through theta.

        :param test_data: 2D ndarray; examples in rows, features in cols.
        :return: int; predicted class index.
        """
        prediction = test_data
        for k in sorted(self.theta.keys()):
            # Add 1s for bias term.
            prediction = np.c_[np.ones([prediction.shape[0], 1]), prediction]
            hypo = prediction @ self.theta[k].transpose()
            prediction = MLFuncts.sigmoid(hypo)

        return np.argmax(prediction, axis=1)

    def get_metrics(self, weighted=True, details=False):
        """Returns f1-score and accuracy metrics.

        :param weighted: bool; If True, 'weighted' score is returned, 'macro' otherwise.
        :param details: bool; prints metrics per class.
        :return: float; f1-score, accuracy
        """
        last_layer = max(self.net.keys())
        predictions = self.net[last_layer]
        max_pred = np.argmax(predictions, axis=1)

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

    Theta1 = np.array([[0.18, 0.44, 0.29, 0.11],
                       [0.33, 0.22, 0.77, 0.88],
                       [0.64, 0.52, 0.11, 0.34]])

    Theta2 = np.array([[0.22, 0.14, 0.11, 0.8],
                       [0.55, 0.33, 0.11, 0.3]])

    Thetas = {0: Theta1, 1: Theta2}

    # test examples
    example1 = np.array([[0.51, 0.26, 0.71]])
    example2 = np.array([[0.51, 0.26, 0.71], [0.48, 0.77, 0.61]])

    # Initialize with weights.
    net1 = Net(X, [3], Y, theta=Thetas)

    net1.train(iterations=3, Lambda=0, method='L-BFGS-B')
    metrics = net1.get_metrics(weighted=False, details=False)
    print("F1-score: {} Accuracy: {}".format(*metrics))

    print(net1.predict(example1))
    print(net1.predict(example2))

    # Initialization without weights (they will be auto computed) and with more layers.
    net2 = Net(X, [30, 20, 5], Y)

    net2.train(iterations=30, Lambda=0, method='L-BFGS-B')
    metrics = net2.get_metrics(weighted=False, details=False)
    print("F1-score: {} Accuracy: {}".format(*metrics))

    print(net2.predict(example1))
    print(net2.predict(example2))
