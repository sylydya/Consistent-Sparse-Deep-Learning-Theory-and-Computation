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




class Drop_out_Net_relu(torch.nn.Module):
    def __init__(self):
        super(Drop_out_Net_relu, self).__init__()
        self.drop1 = nn.Dropout(p = 0.2)
        self.fc1 = nn.Linear(2000, 6)
        self.drop2 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(6,4)
        self.drop3 = nn.Dropout(p = 0.5)
        self.fc3 = nn.Linear(4,3)
        self.drop4 = nn.Dropout(p = 0.5)
        self.fc4 = nn.Linear(3,1)

    def forward(self, x):
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = F.relu(self.fc3(x))
        x = self.drop4(x)
        x = self.fc4(x)
        return x

class Drop_out_Net_tanh(torch.nn.Module):
    def __init__(self):
        super(Drop_out_Net_tanh, self).__init__()
        self.drop1 = nn.Dropout(p = 0.2)
        self.fc1 = nn.Linear(2000, 6)
        self.drop2 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(6,4)
        self.drop3 = nn.Dropout(p = 0.5)
        self.fc3 = nn.Linear(4,3)
        self.drop4 = nn.Dropout(p = 0.5)
        self.fc4 = nn.Linear(3,1)

    def forward(self, x):
        x = self.drop1(x)
        x = F.tanh(self.fc1(x))
        x = self.drop2(x)
        x = F.tanh(self.fc2(x))
        x = self.drop3(x)
        x = F.tanh(self.fc3(x))
        x = self.drop4(x)
        x = self.fc4(x)
        return x


def main():
    my_seed = args.data_index
    subn = 500

    np.random.seed(my_seed - 1)
    torch.manual_seed(my_seed - 1)

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



    temp = np.matrix(pd.read_csv("./data/regression/" + str(my_seed) + "/x_train.csv"))
    x_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(my_seed) + "/y_train.csv"))
    y_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(my_seed) + "/x_val.csv"))
    x_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(my_seed) + "/y_val.csv"))
    y_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(my_seed) + "/x_test.csv"))
    x_test[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/regression/" + str(my_seed) + "/y_test.csv"))
    y_test[:, :] = temp[:, 1:]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    if args.activation == 'tanh':
        net = Drop_out_Net_tanh()
    elif args.activation == 'relu':
        net = Drop_out_Net_relu()
    else:
        print('unrecognized activation function')
        exit(0)
    net.to(device)
    loss_func = nn.MSELoss()

    step_lr = 0.005
    optimization = torch.optim.SGD(net.parameters(), lr=step_lr)

    max_loop = 80001
    PATH = './result/regression/' + args.activation + '/drop_out/'

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

    for para in net.parameters():
        para_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))

    train_loss_path = np.zeros([max_loop // show_information + 1])
    val_loss_path = np.zeros([max_loop // show_information + 1])
    test_loss_path = np.zeros([max_loop // show_information + 1])

    for iter in range(max_loop):
        net.train()
        if subn == NTrain:
            subsample = range(NTrain)
        else:
            subsample = np.random.choice(range(NTrain), size=subn, replace=False)

        net.zero_grad()
        output = net(x_train[subsample,])
        loss = loss_func(output, y_train[subsample,])

        loss.backward()
        optimization.step()


        if iter % show_information == 0:
            net.eval()
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
                for i, para in enumerate(net.parameters()):
                    para_path[i][iter // show_information,] = para.cpu().data.numpy()

    import pickle

    filename = PATH + "simu_" + str(my_seed) + '_drop_out.txt'
    f = open(filename, 'wb')
    pickle.dump([para_path, train_loss_path, val_loss_path, test_loss_path], f)
    f.close()

if __name__ == '__main__':
    main()
