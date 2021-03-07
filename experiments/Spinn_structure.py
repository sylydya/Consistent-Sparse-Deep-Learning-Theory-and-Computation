import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import pandas as pd
import argparse
import errno

import os

parser = argparse.ArgumentParser(description='Spinn Structure')

# Basic Setting
parser.add_argument('--data_index', default=1, type = int, help = 'set data index')
args = parser.parse_args()


class my_Net(torch.nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()
        self.gamma = []

        self.fc1 = nn.Linear(1000, 5, bias=False)
        self.gamma.append(torch.ones(self.fc1.weight.shape, dtype=torch.float32))
        self.fc2 = nn.Linear(5,3, bias=False)
        self.gamma.append(torch.ones(self.fc2.weight.shape, dtype=torch.float32))
        self.fc3 = nn.Linear(3,1, bias=False)
        self.gamma.append(torch.ones(self.fc3.weight.shape, dtype=torch.float32))

    def to(self, *args, **kwargs):
        super(my_Net, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for index in range(self.gamma.__len__()):
            self.gamma[index] = self.gamma[index].to(device)


    def forward(self, x):
        for i, para in enumerate(self.parameters()):
            para.data.mul_(self.gamma[i])
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x

    def mask(self, user_gamma, device):
        for i,para in enumerate(self.parameters()):
            if self.gamma[i].shape != user_gamma[i].shape:
                print('size doesn\'t match')
                return 0
        for i, para in enumerate(self.parameters()):
            self.gamma[i].data = torch.tensor(user_gamma[i], dtype = torch.float32).to(device)


def main():
    my_seed = args.data_index
    subn = 500
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)

    NTrain = 10000
    Nval = 1000
    NTest = 1000
    TotalP = 1000

    x_train = np.matrix(np.zeros([NTrain, TotalP]))
    y_train = np.matrix(np.zeros([NTrain, 1]))

    x_val = np.matrix(np.zeros([Nval, TotalP]))
    y_val = np.matrix(np.zeros([Nval, 1]))

    x_test = np.matrix(np.zeros([NTest, TotalP]))
    y_test = np.matrix(np.zeros([NTest, 1]))

    temp = np.matrix(pd.read_csv("./data/structure/" + str(my_seed) + "/x_train.csv"))
    x_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/structure/" + str(my_seed) + "/y_train.csv"))
    y_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/structure/" + str(my_seed) + "/x_val.csv"))
    x_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/structure/" + str(my_seed) + "/y_val.csv"))
    y_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/structure/" + str(my_seed) + "/x_test.csv"))
    x_test[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/structure/" + str(my_seed) + "/y_test.csv"))
    y_test[:, :] = temp[:, 1:]

    # lambda_vec = np.array([0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
    lambda_vec = np.array([0.05])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    threshold = 0.01

    num_selection_list_group_lasso = np.zeros([lambda_vec.shape[0]])
    num_selection_true_list_group_lasso = np.zeros([lambda_vec.shape[0]])
    train_loss_list_group_lasso = np.zeros([lambda_vec.shape[0]])
    val_loss_list_group_lasso = np.zeros([lambda_vec.shape[0]])
    test_loss_list_group_lasso = np.zeros([lambda_vec.shape[0]])

    train_loss_list_group_lasso_fine_tune = np.zeros([lambda_vec.shape[0]])
    val_loss_list_group_lasso_fine_tune = np.zeros([lambda_vec.shape[0]])
    test_loss_list_group_lasso_fine_tune = np.zeros([lambda_vec.shape[0]])

    for hyper_index in range(lambda_vec.shape[0]):
        np.random.seed(my_seed - 1)
        torch.manual_seed(my_seed - 1)
        lambda_1 = lambda_vec[hyper_index]

        net = my_Net()
        net.to(device)
        loss_func = nn.MSELoss()

        step_lr = 0.005
        optimization = torch.optim.SGD(net.parameters(), lr=step_lr)

        alpha = 0


        max_loop = 80001
        PATH = './result/structure/Spinn/'
        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        show_information = 100

        para_path_group_lasso = []
        para_gamma_path_group_lasso = []

        for para in net.parameters():
            para_path_group_lasso.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))
            para_gamma_path_group_lasso.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))

        train_loss_path_group_lasso = np.zeros([max_loop // show_information + 1])
        val_loss_path_group_lasso = np.zeros([max_loop // show_information + 1])
        test_loss_path_group_lasso = np.zeros([max_loop // show_information + 1])

        for iter in range(max_loop):
            if subn == NTrain:
                subsample = range(NTrain)
            else:
                subsample = np.random.choice(range(NTrain), size=subn, replace=False)

            net.zero_grad()
            output = net(x_train[subsample,])
            loss = loss_func(output, y_train[subsample,])

            penalty = 0
            for i, para in enumerate(net.parameters()):
                penalty = penalty + para.abs().sum().mul((1 - alpha) * lambda_1)

            object = loss + penalty
            object.backward()
            optimization.step()

            if iter % show_information == 0:
                print('iteration:', iter)
                print('lambda1:', lambda_1)
                with torch.no_grad():
                    output = net(x_train)
                    loss = loss_func(output, y_train)
                    print("train loss:", loss)
                    train_loss_path_group_lasso[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_val)
                    loss = loss_func(output, y_val)
                    print("val loss:", loss)
                    val_loss_path_group_lasso[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_test)
                    loss = loss_func(output, y_test)
                    print("test loss:", loss)
                    test_loss_path_group_lasso[iter // show_information] = loss.cpu().data.numpy()

                    for i, para in enumerate(net.parameters()):
                        para_path_group_lasso[i][iter // show_information,] = para.cpu().data.numpy()
                        para_gamma_path_group_lasso[i][iter // show_information,] = (
                                para.abs() > threshold).cpu().data.numpy()

                    print('number of 1:',
                          np.sum(np.max(para_gamma_path_group_lasso[0][iter // show_information,], 0) > 0))
                    print('number of true:',
                          np.sum((np.max(para_gamma_path_group_lasso[0][iter // show_information,], 0) > 0)[0:5]))

        import pickle

        filename = PATH + 'simu_' + str(my_seed) + '_group_lasso_' + str(lambda_1) + '.txt'
        f = open(filename, 'wb')
        pickle.dump(
            [para_path_group_lasso, para_gamma_path_group_lasso, train_loss_path_group_lasso, val_loss_path_group_lasso,
             test_loss_path_group_lasso], f)
        f.close()

        num_selection_list_group_lasso[hyper_index] = np.sum(np.max(para_gamma_path_group_lasso[0][-1,], 0) > 0)
        num_selection_true_list_group_lasso[hyper_index] = np.sum(
            (np.max(para_gamma_path_group_lasso[0][-1,], 0) > 0)[0:5])

        user_gamma = []
        for index in range(para_gamma_path_group_lasso.__len__()):
            user_gamma.append(para_gamma_path_group_lasso[index][-1,])

        with torch.no_grad():
            for i, para in enumerate(net.parameters()):
                para.data = torch.FloatTensor(para_path_group_lasso[i][-1,]).to(device)

        net.mask(user_gamma, device)

        output = net(x_train)
        loss = loss_func(output, y_train)
        print("Train Loss:", loss)
        train_loss_list_group_lasso[hyper_index] = loss.cpu().data.numpy()

        output = net(x_val)
        loss = loss_func(output, y_val)
        print("Val Loss:", loss)
        val_loss_list_group_lasso[hyper_index] = loss.cpu().data.numpy()

        output = net(x_test)
        loss = loss_func(output, y_test)
        print("Test Loss:", loss)
        test_loss_list_group_lasso[hyper_index] = loss.cpu().data.numpy()

        fine_tune_loop = 40001
        para_path_fine_tune_group_lasso = []
        para_gamma_path_fine_tune_group_lasso = []

        for para in net.parameters():
            para_path_fine_tune_group_lasso.append(
                np.zeros([fine_tune_loop // show_information + 1] + list(para.shape)))
            para_gamma_path_fine_tune_group_lasso.append(
                np.zeros([fine_tune_loop // show_information + 1] + list(para.shape)))

        train_loss_path_fine_tune_group_lasso = np.zeros([fine_tune_loop // show_information + 1])
        val_loss_path_fine_tune_group_lasso = np.zeros([fine_tune_loop // show_information + 1])
        test_loss_path_fine_tune_group_lasso = np.zeros([fine_tune_loop // show_information + 1])


        for iter in range(fine_tune_loop):
            if subn == NTrain:
                subsample = range(NTrain)
            else:
                subsample = np.random.choice(range(NTrain), size=subn, replace=False)

            net.zero_grad()
            output = net(x_train[subsample,])
            loss = loss_func(output, y_train[subsample,])

            penalty = 0
            for i, para in enumerate(net.parameters()):
                penalty = penalty + para.abs().sum().mul((1 - alpha) * lambda_1)

            object = loss + penalty
            object.backward()
            optimization.step()

            if iter % show_information == 0:
                print('iteration:', iter)
                with torch.no_grad():
                    output = net(x_train)
                    loss = loss_func(output, y_train)
                    print("train loss:", loss)
                    train_loss_path_fine_tune_group_lasso[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_val)
                    loss = loss_func(output, y_val)
                    print("val loss:", loss)
                    val_loss_path_fine_tune_group_lasso[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_test)
                    loss = loss_func(output, y_test)
                    print("test loss:", loss)
                    test_loss_path_fine_tune_group_lasso[iter // show_information] = loss.cpu().data.numpy()

                    for i, para in enumerate(net.parameters()):
                        para_path_fine_tune_group_lasso[i][iter // show_information,] = para.cpu().data.numpy()
                        para_gamma_path_fine_tune_group_lasso[i][iter // show_information,] = (
                                para.abs() > threshold).cpu().data.numpy()

                    print('number of 1:',
                          np.sum(np.max(para_gamma_path_fine_tune_group_lasso[0][iter // show_information,], 0) > 0))
                    print('number of true:',
                          np.sum((np.max(para_gamma_path_fine_tune_group_lasso[0][iter // show_information,], 0) > 0)[
                                 0:5]))

        import pickle

        filename = PATH + 'simu_' + str(my_seed) + '_group_lasso_fine_tune_' + str(hyper_index) + '.txt'
        f = open(filename, 'wb')
        pickle.dump([para_path_fine_tune_group_lasso, para_gamma_path_fine_tune_group_lasso,
                     train_loss_path_fine_tune_group_lasso, val_loss_path_fine_tune_group_lasso,
                     test_loss_path_fine_tune_group_lasso], f)
        f.close()

        output = net(x_train)
        loss = loss_func(output, y_train)
        print("Train Loss:", loss)
        train_loss_list_group_lasso_fine_tune[hyper_index] = loss.cpu().data.numpy()

        output = net(x_val)
        loss = loss_func(output, y_val)
        print("Val Loss:", loss)
        val_loss_list_group_lasso_fine_tune[hyper_index] = loss.cpu().data.numpy()

        output = net(x_test)
        loss = loss_func(output, y_test)
        print("Test Loss:", loss)
        test_loss_list_group_lasso_fine_tune[hyper_index] = loss.cpu().data.numpy()

    import pickle

    filename = PATH + 'simu_' + str(my_seed) + '_group_lasso_result.txt'
    f = open(filename, 'wb')
    pickle.dump([num_selection_list_group_lasso, num_selection_true_list_group_lasso, train_loss_list_group_lasso,
                 val_loss_list_group_lasso,
                 test_loss_list_group_lasso, train_loss_list_group_lasso_fine_tune, val_loss_list_group_lasso_fine_tune,
                 test_loss_list_group_lasso_fine_tune], f)
    f.close()


if __name__ == '__main__':
    main()
