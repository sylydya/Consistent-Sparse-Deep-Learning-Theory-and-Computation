import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def preprocess_data(data_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if data_name == 'Simulation':
        a = 1
        b = 1
        TotalP = 2000
        print('p = ', TotalP)
        NTrain = 10000
        x_train = np.matrix(np.zeros([NTrain, TotalP]))
        y_train = np.matrix(np.zeros([NTrain, 1]))

        sigma = 1.0
        for i in range(NTrain):
            if i % 1000 == 0:
                print("x_train generate = ", i)
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            while ee > 10 or ee < -10:
                ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                zj = np.sqrt(sigma) * np.random.normal(0, 1)
                while zj > 10 or zj < -10:
                    zj = np.sqrt(sigma) * np.random.normal(0, 1)
                x_train[i, j] = (a * ee + b * zj) / np.sqrt(a * a + b * b)
            x0 = x_train[i, 0]
            x1 = x_train[i, 1]
            x2 = x_train[i, 2]
            x3 = x_train[i, 3]
            x4 = x_train[i, 4]

            y_train[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)


        NTest = 1000
        x_test = np.mat(np.zeros([NTest, TotalP]))
        y_test = np.mat(np.zeros([NTest, 1]))

        for i in range(NTest):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            while ee > 10 or ee < -10:
                ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                zj = np.sqrt(sigma) * np.random.normal(0, 1)
                while zj > 10 or zj < -10:
                    zj = np.sqrt(sigma) * np.random.normal(0, 1)
                x_test[i, j] = (a * ee + b * zj) / np.sqrt(a * a + b * b)
            x0 = x_test[i, 0]
            x1 = x_test[i, 1]
            x2 = x_test[i, 2]
            x3 = x_test[i, 3]
            x4 = x_test[i, 4]

            y_test[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)


    return x_train, y_train, x_test, y_test

