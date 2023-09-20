import numpy as np
# from torch import nn
import sympy as sp
from sympy import lambdify
from sympy.tensor.array import derive_by_array


class ANN():
    def __init__(self, dim=(1, 2, 1), lr=0.01):
        # super().__init__()

        self.dims = dim

        # Learning rate
        self.learning_rate = lr

        self.weights = []
        self.W = []
        for i in range(len(self.dims) - 1):
            rand_weight = np.random.rand(self.dims[i + 1], self.dims[i] + 1) - 0.5
            #                  ^ output dim    ^ input dim plus bias dim
            # W = (W.T/np.sum(W, axis=1)).T # normalize ROWS for mid-range output

            self.weights.append(rand_weight)

            m, n = self.dims[i + 1], self.dims[i] + 1

            layer_w = []
            for j in range(1, m + 1):
                unit_w = []
                for k in range(1, n + 1):
                    unit_w.append(sp.Symbol(f'w{i + 1}{j}{k}'))  # i - layer, j - unit, k - weight
                layer_w.append(unit_w)

            self.W.append(layer_w)

        self.Y = sp.Symbol('y')
        self.X = []

        for i in range(1, self.dims[0] + 1):
            self.X.append([sp.Symbol(f"x{i}")])

        self.flattened_weights = []

        for layer in self.W:
            for lst in layer:
                self.flattened_weights.extend(lst)

        print(f"Flattened weights: {self.flattened_weights}")

        self.f = self.error(self.X, self.Y, self.W)

        self.gradient_eq = derive_by_array(self.f, self.flattened_weights)
        print(f"Gradient equation: {self.gradient_eq} \nShape: {self.gradient_eq.shape}")

        self.hessian_eq = derive_by_array(derive_by_array(self.f, self.flattened_weights), self.flattened_weights)
        print(f"Hessian equation: {self.hessian_eq} | \nShape: {self.hessian_eq.shape}")

        self.gradient_general = lambdify(self.flattened_weights + self.X + [self.Y], self.gradient_eq, "numpy")
        self.hessian_general = lambdify(self.flattened_weights + self.X + [self.Y], self.hessian_eq, "numpy")

    def error(self, X, labels, W):

        Y = X

        for i in range(len(W)):

            # Adding bias
            temp = [[1]]
            temp.extend(Y)
            Y = temp
            X = Y

            Y = []
            for j in range(len(W[i])):
                unit_weight = W[i][j]
                unit_product = []
                for k in range(len(unit_weight)):
                    unit_product.append(unit_weight[k] * X[k][0])

                unit_summation = 0

                for product in unit_product:
                    unit_summation += product

                Y.append([unit_summation])

        print(f"Final Output equation: {Y}")

        return (labels - Y[0][0]) ** 2

    def my_error(self, X, labels):

        Y = [X]

        for i in range(len(self.weights)):

            # Adding bias
            temp = [[1]]
            temp.extend(Y)
            Y = temp
            X = Y

            Y = []
            for j in range(len(self.weights[i])):
                unit_weight = self.weights[i][j]
                unit_product = []
                for k in range(len(unit_weight)):
                    temp = X[k][0]
                    unit_product.append(unit_weight[k] * temp)

                unit_summation = 0

                for product in unit_product:
                    unit_summation += product

                Y.append([unit_summation])

        return (labels - Y[0][0]) ** 2

    def predict(self, testX):

        result = []
        for x in testX:
            layer1 = []
            for weight in self.w1:
                layer1.append(weight * x)

            prediction = 0

            for weight, hidden_val in zip(self.w2, layer1):
                prediction += (weight * hidden_val)

            result.append(prediction)

        return np.array(result)

    def train(self, trainX, trainY, epochs):

        for epoch in range(epochs):

            print(f"--------------- EPOCH {epoch + 1} STARTS ---------------")
            flat_weights = []

            for layer in self.weights:
                for lst in layer:
                    flat_weights.extend(lst)

            # print(np.array(flat_weights))

            grad = lambda ex, why: self.gradient_general(*flat_weights, ex, why)
            hess = lambda ex, why: self.hessian_general(*flat_weights, ex, why)

            final_grad = np.zeros_like(self.gradient_eq)
            final_hess = np.zeros_like(self.hessian_eq)

            # print(final_hess)

            epoch_error = 0
            for i, j in zip(trainX, trainY):
                final_grad = np.add(final_grad, np.array(grad(ex=i, why=j)).squeeze())
                final_hess = np.add(final_hess, np.array(hess(ex=i, why=j)).squeeze())

                epoch_error += self.my_error(i, j)

            print(f"Average error: {epoch_error / len(trainX)}")

            final_grad /= len(trainX)
            final_hess /= len(trainX)

            for i in range(len(final_hess)):
                for j in range(len(final_hess[0])):
                    if isinstance(final_hess[i][j], np.ndarray):
                        final_hess[i][j] = final_hess[i][j][0]

            # print(f"Final grad: {final_grad.shape}")
            # print(f"Final hess: {final_hess.shape}")

            gradient_matrix = final_grad.reshape(-1, 1)

            # return final_hess

            final_hess = final_hess.tolist()
            final_hess = np.array(final_hess)

            hess_inv = np.linalg.inv(final_hess)

            # print(f"Hessian inverse shape: {hess_inv.shape}")

            update = np.matmul(hess_inv, gradient_matrix).squeeze()

            weight_idx = 0
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weights[i][j][k] -= (self.learning_rate) * update[weight_idx]
                        # print(weight_idx)
                        weight_idx += 1


