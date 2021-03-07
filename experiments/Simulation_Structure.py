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


def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    '''
    Compute the gradient of `outputs` with respect to `inputs`
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])

def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    '''
    Compute the Hessian of `output` with respect to `inputs`
    hessian((x * y).sum(), [x, y])
    '''
    #assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out



def main():
    data_index = args.data_index
    subn = 1000

    prior_sigma_0 = 0.0005

    lambda_n = 0.00001

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

    temp = np.matrix(pd.read_csv(
        './data/structure/' + str(data_index) + "/x_train.csv"))
    x_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv(
        './data/structure/' + str(data_index) + "/y_train.csv"))
    y_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv(
        './data/structure/' + str(data_index) + "/x_val.csv"))
    x_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv(
        './data/structure/' + str(data_index) + "/y_val.csv"))
    y_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv(
        './data/structure/' + str(data_index) + "/x_test.csv"))
    x_test[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv(
        './data/structure/' + str(data_index) + "/y_test.csv"))
    y_test[:, :] = temp[:, 1:]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    num_seed = 10

    selected_hessian_list = []

    hessian_part_list = np.zeros([num_seed])

    log_evidence_list = np.zeros([num_seed])
    dim_list = np.zeros([num_seed])
    num_selection_list = np.zeros([num_seed])
    num_selection_true_list = np.zeros([num_seed])
    train_loss_list = np.zeros([num_seed])
    val_loss_list = np.zeros([num_seed])
    test_loss_list = np.zeros([num_seed])



    for my_seed in range(num_seed):
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)

        net = my_Net()
        net.to(device)
        loss_func = nn.MSELoss()

        step_lr = 0.01
        optimization = torch.optim.SGD(net.parameters(), lr=step_lr)

        sigma = torch.FloatTensor([1]).to(device)


        prior_sigma_1 = 0.01

        threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
                0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

        c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
        c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1

        max_loop = 80001
        PATH = './result/structure/bayesian_evidence/'

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
        gamma_dist = []
        for para in net.parameters():
            para_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))
            para_gamma_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))
            gamma_dist.append(torch.distributions.Uniform(torch.zeros(para.shape), torch.ones(para.shape)))

        train_loss_path = np.zeros([max_loop // show_information + 1])
        val_loss_path = np.zeros([max_loop // show_information + 1])
        test_loss_path = np.zeros([max_loop // show_information + 1])


        for iter in range(max_loop):
            if subn == NTrain:
                subsample = range(NTrain)
            else:
                subsample = np.random.choice(range(NTrain), size=subn, replace=False)

            net.zero_grad()
            output = net(x_train[subsample,])
            loss = loss_func(output, y_train[subsample,])
            loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))

            loss.backward()

            # prior gradient
            with torch.no_grad():
                for para in net.parameters():
                    temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                    prior_grad = temp.div(NTrain)
                    para.grad.data -= prior_grad

            optimization.step()

            # with torch.no_grad():
            #     sigma.data = sigma.sub(sigma.grad.mul(step_lr)).data
            # sigma.grad.zero_()

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
                        para_gamma_path[i][iter // show_information,] = (para.abs() > threshold).cpu().data.numpy()

                    print('number of 1:', np.sum(np.max(para_gamma_path[0][iter // show_information,], 0) > 0))
                    print('number of true:',
                          np.sum((np.max(para_gamma_path[0][iter // show_information,], 0) > 0)[0:5]))
                    print('threshold = :', threshold)
                    print(para_gamma_path[0][iter // show_information,][:, 0:5])
                    print(para_gamma_path[1][iter // show_information,])
                    print(para_gamma_path[2][iter // show_information,])

        import pickle

        filename = PATH + 'data_' + str(data_index) + "_simu_" + str(my_seed) + '_' + str(subn) + '_' + str(
            lambda_n) + '_' + str(prior_sigma_0) + '.txt'
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

            # prior gradient
            with torch.no_grad():
                for para in net.parameters():
                    temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                    prior_grad = temp.div(NTrain)
                    para.grad.data -= prior_grad

            optimization.step()

            # with torch.no_grad():
            #     sigma.data = sigma.sub(sigma.grad.mul(step_lr)).data
            # sigma.grad.zero_()

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
                        para_gamma_path_fine_tune[i][iter // show_information,] = (
                                    para.abs() > threshold).cpu().data.numpy()

                    print('number of 1:',
                          np.sum(np.max(para_gamma_path_fine_tune[0][iter // show_information,], 0) > 0))
                    print('number of true:',
                          np.sum((np.max(para_gamma_path_fine_tune[0][iter // show_information,], 0) > 0)[0:5]))

        import pickle

        filename = PATH + 'data_' + str(data_index) + "_simu_" + str(my_seed) + '_' + str(subn) + '_' + str(
            lambda_n) + '_' + str(
            prior_sigma_0) + '_fine_tune.txt'
        f = open(filename, 'wb')
        pickle.dump([para_path_fine_tune, para_gamma_path_fine_tune, train_loss_path_fine_tune, val_loss_path_fine_tune,
                     test_loss_path_fine_tune], f)
        f.close()

        output = net(x_train)
        loss = loss_func(output, y_train)
        loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))
        # loss = loss.mul(NTrain)

        prior = 0
        for layer_index, para in enumerate(net.parameters()):
            temp = (para.pow(2).div(-2 * prior_sigma_1).exp().mul(lambda_n / np.sqrt(prior_sigma_1)) +
                    para.pow(2).div(-2 * prior_sigma_0).exp().mul(
                        (1 - lambda_n) / np.sqrt(prior_sigma_0))).log()
            debug_temp = para.pow(2).div(-2 * prior_sigma_1).add(np.log(lambda_n / np.sqrt(prior_sigma_1)))
            temp = torch.where(torch.isinf(temp), debug_temp, temp)
            prior = prior - temp.mul(net.gamma[layer_index]).sum()


        prior = prior.div(NTrain)
        object = loss + prior
        loss_hessian = hessian(loss, net.parameters())
        prior_hessian = hessian(prior, net.parameters())

        shape = prior_hessian.shape[0]

        debug_hessian = torch.eye(shape).mul(prior_sigma_1 / NTrain).to(device)

        prior_hessian = torch.where(torch.isnan(prior_hessian), debug_hessian, prior_hessian)

        hessian_matrix = loss_hessian + prior_hessian

        gamma_index = torch.cat([x.contiguous().view(-1) for x in net.gamma])

        selected_hessian = hessian_matrix[np.where(gamma_index.cpu().numpy() > 0.5)[0], :][:,
                           np.where(gamma_index.cpu().numpy() > 0.5)[0]]

        dim = np.where(gamma_index.cpu().numpy() > 0.5)[0].shape[0]

        hessian_part = 0.5 * dim * np.log(2 * np.pi) - 0.5 * dim * np.log(NTrain) - 0.5 * \
                       selected_hessian.eig()[0][:, 0].abs().log().sum()
        log_evidence = object.mul(-NTrain) + hessian_part

        print('Bayesian Evidence:', log_evidence)
        print('Dimension:', dim)
        log_evidence_list[my_seed] = log_evidence.cpu().data.numpy()
        dim_list[my_seed] = dim
        hessian_part_list[my_seed] = hessian_part
        selected_hessian_list.append(selected_hessian)

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

        user_gamma = []
        for index in range(para_gamma_path_fine_tune.__len__()):
            user_gamma.append(para_gamma_path_fine_tune[index][-1,])

        with torch.no_grad():
            for i, para in enumerate(net.parameters()):
                para.data = torch.FloatTensor(para_path_fine_tune[i][-1,]).to(device)

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

            # prior gradient
            with torch.no_grad():
                for para in net.parameters():
                    temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                    prior_grad = temp.div(NTrain)
                    para.grad.data -= prior_grad

            optimization.step()

            # with torch.no_grad():
            #     sigma.data = sigma.sub(sigma.grad.mul(step_lr)).data
            # sigma.grad.zero_()

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
                        para_gamma_path_fine_tune[i][iter // show_information,] = (
                                    para.abs() > threshold).cpu().data.numpy()

                    # print(net.fc1.weight[0, 0])
                    # print(para_path_fine_tune[0][iter // show_information, 0, 0])
                    print('number of 1:',
                          np.sum(np.max(para_gamma_path_fine_tune[0][iter // show_information,], 0) > 0))
                    print('number of true:',
                          np.sum((np.max(para_gamma_path_fine_tune[0][iter // show_information,], 0) > 0)[0:5]))

        import pickle

        filename = PATH + 'data_' + str(data_index) + "_simu_" + str(my_seed) + '_' + str(subn) + '_' + str(
            lambda_n) + '_' + str(
            prior_sigma_0) + '_fine_tune_again.txt'
        f = open(filename, 'wb')
        pickle.dump([para_path_fine_tune, para_gamma_path_fine_tune, train_loss_path_fine_tune, val_loss_path_fine_tune,
                     test_loss_path_fine_tune], f)
        f.close()

        output = net(x_train)
        loss = loss_func(output, y_train)
        loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))
        # loss = loss.mul(NTrain)

        prior = 0
        for layer_index, para in enumerate(net.parameters()):
            temp = (para.pow(2).div(-2 * prior_sigma_1).exp().mul(lambda_n / np.sqrt(prior_sigma_1)) +
                    para.pow(2).div(-2 * prior_sigma_0).exp().mul(
                        (1 - lambda_n) / np.sqrt(prior_sigma_0))).log()
            debug_temp = para.pow(2).div(-2 * prior_sigma_1).add(np.log(lambda_n / np.sqrt(prior_sigma_1)))
            temp = torch.where(torch.isinf(temp), debug_temp, temp)
            prior = prior - temp.mul(net.gamma[layer_index]).sum()

        # prior = 0
        # for layer_index, para in enumerate(net.parameters()):
        #     prior = prior - (para.pow(2).div(-2 * prior_sigma_1).exp().mul(lambda_n / np.sqrt(prior_sigma_1)) +
        #                      para.pow(2).div(-2 * prior_sigma_0).exp().mul(
        #                          (1 - lambda_n) / np.sqrt(prior_sigma_0))).log().mul(net.gamma[layer_index]).sum()
        #     # print(prior)

        # prior = prior + sigma.log().mul(inverse_gamma_alpha + 1) + sigma.reciprocal().mul(inverse_gamma_beta)
        prior = prior.div(NTrain)
        object = loss + prior

        loss_hessian = hessian(loss, net.parameters())
        prior_hessian = hessian(prior, net.parameters())

        shape = prior_hessian.shape[0]

        debug_hessian = torch.eye(shape).mul(prior_sigma_1 / NTrain).to(device)

        prior_hessian = torch.where(torch.isnan(prior_hessian), debug_hessian, prior_hessian)

        hessian_matrix = loss_hessian + prior_hessian

        gamma_index = torch.cat([x.contiguous().view(-1) for x in net.gamma])

        selected_hessian = hessian_matrix[np.where(gamma_index.cpu().numpy() > 0.5)[0], :][:,
                           np.where(gamma_index.cpu().numpy() > 0.5)[0]]

        dim = np.where(gamma_index.cpu().numpy() > 0.5)[0].shape[0]

        hessian_part = 0.5 * dim * np.log(2 * np.pi) - 0.5 * dim * np.log(NTrain) - 0.5 * \
                       selected_hessian.eig()[0][:, 0].abs().log().sum()
        log_evidence = object.mul(-NTrain) + hessian_part

        # log_evidence = object.mul(-NTrain) + 0.5 * dim * np.log(2 * np.pi) - 0.5 * dim * np.log(NTrain) - 0.5 * \
        #                selected_hessian.eig()[0][:, 0].abs().log().sum()

        print('Bayesian Evidence:', log_evidence)
        print('Dimension:', dim)

        log_evidence_list_fine_tune[my_seed] = log_evidence.cpu().data.numpy()
        dim_list_fine_tune[my_seed] = dim
        hessian_part_list_fine_tune[my_seed] = hessian_part
        selected_hessian_list_fine_tune.append(selected_hessian)

        output = net(x_train)
        loss = loss_func(output, y_train)
        print("Train Loss:", loss)
        train_loss_list_fine_tune[my_seed] = loss.cpu().data.numpy()

        output = net(x_val)
        loss = loss_func(output, y_val)
        print("Val Loss:", loss)
        val_loss_list_fine_tune[my_seed] = loss.cpu().data.numpy()

        output = net(x_test)
        loss = loss_func(output, y_test)
        print("Test Loss:", loss)
        test_loss_list_fine_tune[my_seed] = loss.cpu().data.numpy()

    import pickle

    filename = PATH + 'data_' + str(data_index) + '_result.txt'
    f = open(filename, 'wb')
    # pickle.dump([log_evidence_list, dim_list, num_selection_list, num_selection_true_list, train_loss_list, val_loss_list ,test_loss_list,
    #              log_evidence_list_fine_tune, dim_list_fine_tune, num_selection_list_fine_tune, num_selection_true_list_fine_tune, train_loss_list_fine_tune, val_loss_list_fine_tune, test_loss_list_fine_tune], f)
    pickle.dump([log_evidence_list, dim_list, hessian_part_list, selected_hessian_list, num_selection_list,
                 num_selection_true_list, train_loss_list, val_loss_list, test_loss_list,
                 log_evidence_list_fine_tune, dim_list_fine_tune, hessian_part_list_fine_tune,
                 selected_hessian_list_fine_tune, num_selection_list_fine_tune, num_selection_true_list_fine_tune,
                 train_loss_list_fine_tune, val_loss_list_fine_tune, test_loss_list_fine_tune], f)

    f.close()

if __name__ == '__main__':
    main()

