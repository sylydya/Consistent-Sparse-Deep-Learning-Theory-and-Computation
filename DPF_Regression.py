import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import argparse
import errno

import os

parser = argparse.ArgumentParser(description='Simulation Regression')

# Basic Setting
parser.add_argument('--data_index', default=1, type = int, help = 'set data index')
parser.add_argument('--activation', default='tanh', type = str, help = 'set activation function')
args = parser.parse_args()

class my_Net_tanh(torch.nn.Module):
    def __init__(self):
        super(my_Net_tanh, self).__init__()
        self.gamma = []
        self.fc1 = nn.Linear(2000, 6)
        self.gamma.append(torch.ones(self.fc1.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc1.bias.shape, dtype=torch.float32))
        self.fc2 = nn.Linear(6,4)
        self.gamma.append(torch.ones(self.fc2.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc2.bias.shape, dtype=torch.float32))
        self.fc3 = nn.Linear(4,3)
        self.gamma.append(torch.ones(self.fc3.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc3.bias.shape, dtype=torch.float32))
        self.fc4 = nn.Linear(3,1)
        self.gamma.append(torch.ones(self.fc4.weight.shape, dtype = torch.float32))
        self.gamma.append(torch.ones(self.fc4.bias.shape, dtype = torch.float32))

    def to(self, *args, **kwargs):
        super(my_Net_tanh, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for index in range(self.gamma.__len__()):
            self.gamma[index] = self.gamma[index].to(device)

    def forward(self, x):
        for i, para in enumerate(self.parameters()):
            para.data.mul_(self.gamma[i])
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

    def mask(self, user_gamma, device):
        for i,para in enumerate(self.parameters()):
            if self.gamma[i].shape != user_gamma[i].shape:
                print('size doesn\'t match')
                return 0
        for i, para in enumerate(self.parameters()):
            self.gamma[i].data = torch.tensor(user_gamma[i], dtype = torch.float32).to(device)

class my_Net_relu(torch.nn.Module):
    def __init__(self):
        super(my_Net_relu, self).__init__()
        self.gamma = []
        self.fc1 = nn.Linear(2000, 6)
        self.gamma.append(torch.ones(self.fc1.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc1.bias.shape, dtype=torch.float32))
        self.fc2 = nn.Linear(6,4)
        self.gamma.append(torch.ones(self.fc2.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc2.bias.shape, dtype=torch.float32))
        self.fc3 = nn.Linear(4,3)
        self.gamma.append(torch.ones(self.fc3.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc3.bias.shape, dtype=torch.float32))
        self.fc4 = nn.Linear(3,1)
        self.gamma.append(torch.ones(self.fc4.weight.shape, dtype = torch.float32))
        self.gamma.append(torch.ones(self.fc4.bias.shape, dtype = torch.float32))

    def to(self, *args, **kwargs):
        super(my_Net_relu, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for index in range(self.gamma.__len__()):
            self.gamma[index] = self.gamma[index].to(device)

    def forward(self, x):
        for i, para in enumerate(self.parameters()):
            para.data.mul_(self.gamma[i])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def mask(self, user_gamma, device):
        for i,para in enumerate(self.parameters()):
            if self.gamma[i].shape != user_gamma[i].shape:
                print('size doesn\'t match')
                return 0
        for i, para in enumerate(self.parameters()):
            self.gamma[i].data = torch.tensor(user_gamma[i], dtype = torch.float32).to(device)


def main():
    data_index = args.data_index
    subn = 500


    NTrain = 10000
    Nval = 1000
    NTest = 1000
    TotalP = 2000

    x_train = np.matrix(np.zeros([NTrain, TotalP]))
    y_train = np.matrix(np.zeros([NTrain, 1]))

    x_val = np.matrix(np.zeros([Nval, TotalP]))
    y_val = np.matrix(np.zeros([Nval, 1]))

    x_test = np.matrix(np.zeros([NTest, TotalP]))
    y_test = np.matrix(np.zeros([NTest, 1]))

    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/x_train.csv"))
    x_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/y_train.csv"))
    y_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/x_val.csv"))
    x_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/y_val.csv"))
    y_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/x_test.csv"))
    x_test[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(data_index) + "/y_test.csv"))
    y_test[:, :] = temp[:, 1:]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    num_seed = 1

    num_selection_list = np.zeros([num_seed])
    num_selection_true_list = np.zeros([num_seed])
    train_loss_list = np.zeros([num_seed])
    val_loss_list = np.zeros([num_seed])
    test_loss_list = np.zeros([num_seed])

    for my_seed in range(num_seed):
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)

        if args.activation == 'tanh':
            net = my_Net_tanh()
            net_dense = my_Net_tanh()
        elif args.activation == 'relu':
            net = my_Net_relu()
            net_dense = my_Net_relu()
        else:
            print('unrecognized activation function')
            exit(0)


        net.load_state_dict(net_dense.state_dict())

        net_dense.to(device)
        net.to(device)
        loss_func = nn.MSELoss()

        step_lr = 0.005
        optimization = torch.optim.SGD(net.parameters(), lr=step_lr, weight_decay=5e-4)
        optimization_dense = torch.optim.SGD(net_dense.parameters(), lr=step_lr, weight_decay=5e-4)

        sigma = torch.FloatTensor([1]).to(device)


        max_loop = 80001
        PATH = './result/regression/' + args.activation + '/DPF/'

        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        show_information = 100

        para_path = []
        para_gamma_path = []
        for para in net.parameters():
            para_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))
            para_gamma_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))

        train_loss_path = np.zeros([max_loop // show_information + 1])
        val_loss_path = np.zeros([max_loop // show_information + 1])
        test_loss_path = np.zeros([max_loop // show_information + 1])



        total_num_para = 0
        for name, para in net.named_parameters():
            total_num_para += para.numel()
        para_placeholder = torch.zeros([total_num_para]).to(device)

        s_init = 0
        s_target = 1 - (total_num_para - (TotalP - 5) * net.fc1.weight.shape[0])/total_num_para


        for iter in range(max_loop):

            if iter < (0.75*max_loop):
                if iter % 16 == 0:

                    sparsity_level = s_target + (s_init - s_target) * np.power(
                        (1 - iter * 1.0 / (0.75*max_loop)), 3)
                    cut_index = int(np.floor((sparsity_level) * (total_num_para - 1)))
                    para_count = 0
                    for para in net_dense.parameters():
                        temp_num = para.numel()
                        para_placeholder[para_count:(para_count + temp_num)] = para.abs().view(-1)
                        para_count = para_count + temp_num

                    threshold = para_placeholder.sort()[0][cut_index]
                    user_gamma = []
                    for i, para in enumerate(net_dense.parameters()):
                        user_gamma.append((para.abs() > threshold).cpu().data.numpy())
                    net.mask(user_gamma, device)


            if subn == NTrain:
                subsample = range(NTrain)
            else:
                subsample = np.random.choice(range(NTrain), size=subn, replace=False)

            net.zero_grad()
            output = net(x_train[subsample,])
            loss = loss_func(output, y_train[subsample,])
            loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))

            loss.backward()

            with torch.no_grad():
                for para_dense, para in zip(net_dense.parameters(), net.parameters()):
                    if para_dense.grad is None:
                        para_dense.grad = torch.zeros_like(para.grad)
                    para_dense.grad.data = para.grad

            optimization.step()
            optimization_dense.step()


            if iter % show_information == 0:
                print('iteration:', iter)
                with torch.no_grad():
                    output = net(x_train)
                    loss = loss_func(output, y_train)
                    print("train loss:", loss)
                    train_loss_path[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_val)
                    loss = loss_func(output, y_val)
                    print("val loss:", loss)
                    val_loss_path[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_test)
                    loss = loss_func(output, y_test)
                    print("test loss:", loss)
                    test_loss_path[iter // show_information] = loss.cpu().data.numpy()
                    print('sigma:', sigma)

                    for i, para in enumerate(net.parameters()):
                        para_path[i][iter // show_information,] = para.cpu().data.numpy()
                        para_gamma_path[i][iter // show_information,] = user_gamma[i]


                    print('number of 1:', np.sum(np.max(para_gamma_path[0][iter // show_information,], 0) > 0))
                    print('number of true:',
                          np.sum((np.max(para_gamma_path[0][iter // show_information,], 0) > 0)[0:5]))
                    print('sparsity: ', sparsity_level)
                    print('cut_index: ', cut_index)

        import pickle

        filename = PATH + 'data_' + str(data_index) + "_simu_" + str(my_seed) + '.txt'
        f = open(filename, 'wb')
        pickle.dump([para_path, para_gamma_path, train_loss_path, val_loss_path, test_loss_path], f)
        f.close()

        num_selection_list[my_seed] = np.sum(np.max(para_gamma_path[0][-1,], 0) > 0)
        num_selection_true_list[my_seed] = np.sum((np.max(para_gamma_path[0][-1,], 0) > 0)[0:5])

        user_gamma = []
        for index in range(para_gamma_path.__len__()):
            user_gamma.append(para_gamma_path[index][-1,])

        with torch.no_grad():
            for i, para in enumerate(net.parameters()):
                para.data = torch.FloatTensor(para_path[i][-1,]).to(device)

        net.mask(user_gamma, device)

        fine_tune_loop = 40001
        para_path_fine_tune = []
        para_gamma_path_fine_tune = []

        for para in net.parameters():
            para_path_fine_tune.append(np.zeros([fine_tune_loop // show_information + 1] + list(para.shape)))
            para_gamma_path_fine_tune.append(np.zeros([fine_tune_loop // show_information + 1] + list(para.shape)))

        train_loss_path_fine_tune = np.zeros([fine_tune_loop // show_information + 1])
        val_loss_path_fine_tune = np.zeros([fine_tune_loop // show_information + 1])
        test_loss_path_fine_tune = np.zeros([fine_tune_loop // show_information + 1])

        for iter in range(fine_tune_loop):
            if subn == NTrain:
                subsample = range(NTrain)
            else:
                subsample = np.random.choice(range(NTrain), size=subn, replace=False)

            net.zero_grad()
            output = net(x_train[subsample,])
            loss = loss_func(output, y_train[subsample,])
            loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))

            loss.backward()

            optimization.step()

            if iter % show_information == 0:
                print('iteration:', iter)
                with torch.no_grad():
                    output = net(x_train)
                    loss = loss_func(output, y_train)
                    print("train loss:", loss)
                    train_loss_path_fine_tune[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_val)
                    loss = loss_func(output, y_val)
                    print("val loss:", loss)
                    val_loss_path_fine_tune[iter // show_information] = loss.cpu().data.numpy()
                    output = net(x_test)
                    loss = loss_func(output, y_test)
                    print("test loss:", loss)
                    test_loss_path_fine_tune[iter // show_information] = loss.cpu().data.numpy()
                    print('sigma:', sigma)

                    for i, para in enumerate(net.parameters()):
                        para_path_fine_tune[i][iter // show_information,] = para.cpu().data.numpy()
                        para_gamma_path_fine_tune[i][iter // show_information,] = user_gamma[i]

                    print('number of 1:',
                          np.sum(np.max(para_gamma_path_fine_tune[0][iter // show_information,], 0) > 0))
                    print('number of true:',
                          np.sum((np.max(para_gamma_path_fine_tune[0][iter // show_information,], 0) > 0)[0:5]))


        import pickle

        filename = PATH + 'data_' + str(data_index) + "_simu_" + str(my_seed)  + '_fine_tune.txt'
        f = open(filename, 'wb')
        pickle.dump([para_path_fine_tune, para_gamma_path_fine_tune, train_loss_path_fine_tune, val_loss_path_fine_tune,
                     test_loss_path_fine_tune], f)
        f.close()


        output = net(x_train)
        loss = loss_func(output, y_train)
        print("Train Loss:", loss)
        train_loss_list[my_seed] = loss.cpu().data.numpy()

        output = net(x_val)
        loss = loss_func(output, y_val)
        print("Val Loss:", loss)
        val_loss_list[my_seed] = loss.cpu().data.numpy()

        output = net(x_test)
        loss = loss_func(output, y_test)
        print("Test Loss:", loss)
        test_loss_list[my_seed] = loss.cpu().data.numpy()

    import pickle

    filename = PATH + 'data_' + str(data_index) + '_result.txt'
    f = open(filename, 'wb')
    pickle.dump([num_selection_list, num_selection_true_list, train_loss_list, val_loss_list, test_loss_list], f)
    f.close()

if __name__ == '__main__':
    main()
