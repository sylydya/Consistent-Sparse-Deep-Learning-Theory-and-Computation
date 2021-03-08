import argparse

import numpy as np
import torch
import torch.nn as nn

import os
import errno

from process_data import preprocess_data

parser = argparse.ArgumentParser(description='BNN with mixture normal prior')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--data_name', default = 'Simulation', type = str, help = 'data name')
parser.add_argument('--base_path', default='./result/', type = str, help = 'base path for saving result')
parser.add_argument('--model_path', default='test_run/', type = str, help = 'folder name for saving model')
parser.add_argument('--fine_tune_path', default='fine_tune/', type = str, help = 'folder name for saving fine tune model')

parser.add_argument('--num_run', default = 1, type = int, help= 'Number of different initialization used to train the model')


# Network Architecture
parser.add_argument('--layer', default=3, type=int, help='number of hidden layer')
parser.add_argument('--unit', default=[6, 4, 3], type=int, nargs='+', help='number of hidden unit in each layer')


# Training Setting
parser.add_argument('--nepoch', default = 500, type = int, help = 'total number of training epochs')
parser.add_argument('--lr', default = 0.005, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0, type = float, help = 'momentum in SGD')
parser.add_argument('--batch_train', default = 100, type = int, help = 'batch size for training')


parser.add_argument('--fine_tune_epoch', default = 500, type = int, help = 'total number of fine tuning epochs')

# Prior Setting
parser.add_argument('--sigma0', default = 0.0001, type = float, help = 'sigma_0^2 in prior')
parser.add_argument('--sigma1', default = 0.01, type = float, help = 'sigma_1^2 in prior')

parser.add_argument('--lambdan', default = 0.00001, type = float, help = 'lambda_n in prior')

args = parser.parse_args()



class Net(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim):
        super(Net, self).__init__()
        self.num_hidden = num_hidden

        self.fc = nn.Linear(input_dim, hidden_dim[0])
        self.fc_list = []

        for i in range(num_hidden - 1):
            self.fc_list.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.add_module('fc' + str(i + 2), self.fc_list[-1])
        self.fc_list.append(nn.Linear(hidden_dim[-1], output_dim))
        self.add_module('fc' + str(num_hidden + 1), self.fc_list[-1])

        self.prune_flag = 0
        self.mask = None

    def forward(self, x):

        if self.prune_flag == 1:
            for name, para in self.named_parameters():
                para.data[self.mask[name]] = 0

        x = torch.tanh(self.fc(x))
        for i in range(self.num_hidden - 1):
            x = torch.tanh(self.fc_list[i](x))
        x = self.fc_list[-1](x)
        return x

    def set_prune(self, user_mask):
        self.mask = user_mask
        self.prune_flag = 1

    def cancel_prune(self):
        self.prune_flag = 0
        self.mask = None



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
    data_name = args.data_name
    subn = args.batch_train

    prior_sigma_0 = args.sigma0

    prior_sigma_1 = args.sigma1

    lambda_n = args.lambdan

    num_hidden = args.layer
    hidden_dim = args.unit

    num_seed = args.num_run

    num_epoch = args.nepoch


    x_train, y_train, x_test, y_test = preprocess_data(data_name)
    output_dim = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ntrain = x_train.shape[0]
    dim = x_train.shape[1]


    dim_list = np.zeros([num_seed])
    BIC_list = np.zeros([num_seed])

    num_selection_list = np.zeros([num_seed])

    train_loss_list = np.zeros([num_seed])
    test_loss_list = np.zeros([num_seed])

    for my_seed in range(num_seed):
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)

        net = Net(num_hidden, hidden_dim, dim, output_dim)
        net.to(device)

        loss_func = nn.MSELoss()

        step_lr = args.lr
        momentum = args.momentum
        optimization = torch.optim.SGD(net.parameters(), lr=step_lr, momentum=momentum)

        sigma = torch.FloatTensor([1]).to(device)

        c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
        c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1
        threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
                0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))


        PATH = args.base_path + args.data_name + '/' + args.model_path

        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        para_path = []
        para_gamma_path = []
        for para in net.parameters():
            para_path.append(np.zeros([num_epoch] + list(para.shape)))
            para_gamma_path.append(np.zeros([num_epoch] + list(para.shape)))

        train_loss_path = np.zeros([num_epoch])
        test_loss_path = np.zeros([num_epoch])

        index = np.arange(ntrain)

        for epoch in range(num_epoch):
            np.random.shuffle(index)

            for iter in range(ntrain // subn):
                subsample = index[(iter * subn):((iter + 1) * subn)]
                net.zero_grad()
                output = net(x_train[subsample,])
                loss = loss_func(output, y_train[subsample,])
                loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))
                loss.backward()
                # prior gradient:
                with torch.no_grad():
                    for para in net.parameters():
                        temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                        temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                        prior_grad = temp.div(ntrain)
                        para.grad.data -= prior_grad
                optimization.step()

            print('epoch:', epoch)
            with torch.no_grad():
                output = net(x_train)
                loss = loss_func(output, y_train)
                print("train loss:", loss)
                train_loss_path[epoch] = loss.cpu().data.numpy()

                output = net(x_test)
                loss = loss_func(output, y_test)
                print("test loss:", loss)
                test_loss_path[epoch] = loss.cpu().data.numpy()

                for i, para in enumerate(net.parameters()):
                    para_path[i][epoch,] = para.cpu().data.numpy()
                    para_gamma_path[i][epoch,] = (para.abs() > threshold).cpu().data.numpy()

                print('number of selected:', np.sum(np.max(para_gamma_path[0][epoch,], 0) > 0))

        import pickle
        filename = PATH + 'result' + str(my_seed)
        f = open(filename, 'wb')
        pickle.dump([para_path, para_gamma_path, train_loss_path, test_loss_path], f)
        f.close()
        num_selection_list[my_seed] = np.sum(np.max(para_gamma_path[0][-1,], 0) > 0)

        temp_str = [str(int(x)) for x in np.max(para_gamma_path[0][-1,], 0) > 0]
        temp_str = ' '.join(temp_str)
        filename = PATH + 'selected_variable' + str(my_seed) + '.txt'
        f = open(filename, 'w')
        f.write(temp_str)
        f.close()


        with torch.no_grad():
            for i, para in enumerate(net.parameters()):
                para.data = torch.FloatTensor(para_path[i][-1,]).to(device)

        user_mask = {}
        for name, para in net.named_parameters():
            user_mask[name] = para.abs() < threshold
        net.set_prune(user_mask)


        fine_tune_epoch = args.fine_tune_epoch
        para_path_fine_tune = []
        para_gamma_path_fine_tune = []

        for para in net.parameters():
            para_path_fine_tune.append(np.zeros([fine_tune_epoch] + list(para.shape)))
            para_gamma_path_fine_tune.append(np.zeros([fine_tune_epoch] + list(para.shape)))

        train_loss_path_fine_tune = np.zeros([fine_tune_epoch])
        test_loss_path_fine_tune = np.zeros([fine_tune_epoch])


        optimization = torch.optim.SGD(net.parameters(), lr=step_lr, momentum=momentum)

        for epoch in range(fine_tune_epoch):

            np.random.shuffle(index)
            for iter in range(ntrain // subn):
                subsample = index[(iter * subn):((iter + 1) * subn)]
                net.zero_grad()
                output = net(x_train[subsample,])
                loss = loss_func(output, y_train[subsample,])
                loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))
                loss.backward()

                # prior gradient:
                with torch.no_grad():
                    for para in net.parameters():
                        temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                        temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                        prior_grad = temp.div(ntrain)
                        para.grad.data -= prior_grad
                optimization.step()

            print('fine tune epoch:', epoch)
            with torch.no_grad():
                output = net(x_train)
                loss = loss_func(output, y_train)
                print("train loss:", loss)
                train_loss_path_fine_tune[epoch] = loss.cpu().data.numpy()

                output = net(x_test)
                loss = loss_func(output, y_test)
                print("test loss:", loss)
                test_loss_path_fine_tune[epoch] = loss.cpu().data.numpy()

                for i, para in enumerate(net.parameters()):
                    para_path_fine_tune[i][epoch,] = para.cpu().data.numpy()
                    para_gamma_path_fine_tune[i][epoch,] = (
                                para.abs() > threshold).cpu().data.numpy()

                print('number of selected:',
                      np.sum(np.max(para_gamma_path_fine_tune[0][epoch,], 0) > 0))

        import pickle

        filename = PATH + 'fine_tune_result' + str(my_seed)
        f = open(filename, 'wb')
        pickle.dump([para_path_fine_tune, para_gamma_path_fine_tune, train_loss_path_fine_tune,
                     test_loss_path_fine_tune], f)
        f.close()


        with torch.no_grad():
            output = net(x_train)
            loss = loss_func(output, y_train)
            print("Train Loss:", loss)
            train_loss_list[my_seed] = loss.cpu().data.numpy()

            num_non_zero_element = 0
            for name, para in net.named_parameters():
                num_non_zero_element = num_non_zero_element + para.numel() - net.mask[name].sum()
            BIC = (2 * ntrain * loss + np.log(ntrain) *num_non_zero_element).item()
            BIC_list[my_seed] = BIC
            dim_list[my_seed] = num_non_zero_element


            output = net(x_test)
            loss = loss_func(output, y_test)
            print("Test Loss:", loss)
            test_loss_list[my_seed] = loss.cpu().data.numpy()
            print("number of non-zero connections:", num_non_zero_element)
            print('BIC:', BIC)

    import pickle

    filename = PATH + 'Overall_result_with_different_initialization'
    f = open(filename, 'wb')
    pickle.dump([BIC_list, dim_list, num_selection_list, train_loss_list, test_loss_list], f)
    f.close()


if __name__ == '__main__':
    main()
