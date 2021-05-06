import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class poly_Model():

    def __init__(self, max_degree=25):
        super().__init__()

    def design_matrix(self, x, degree):
        rows = len(x)
        X = np.zeros((rows, degree+1))
        for i in range(0, rows):
            for j in range(0, degree+1):
                X[i][j] = x[i]**j

        return X

    def train_model(self, x, y, degree):

        X = design_matrix(x, degree)

        alpha = np.dot(np.linalg.pinv(X), y)

        return alpha, X

    def error(self, x, y, alpha, degree):
        X = design_matrix(x, degree)
        y_pred = np.dot(X, alpha)
        err = (np.mean((y_pred-y)**2))

        plt.plot(x, y_pred, 'x')
        plt.title('Model')
        plt.xlabel('support')
        plt.ylabel('y')

        return err, y_pred
