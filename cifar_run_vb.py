import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import torch.utils.data

#import torchvision.transforms as transforms
import transforms
import torchvision.datasets as datasets
import resnet_vb
import os
import errno
from torch.utils.data.sampler import SubsetRandomSampler


parser = argparse.ArgumentParser(description='Cifar10 ResNet Compression')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--base_path', default='./result/cifar/vb/', type = str, help = 'base path for saving result')
parser.add_argument('--model_path', default='test_run_VB/', type = str, help = 'folder name for saving model')
parser.add_argument('--fine_tune_path', default='fine_tune/', type = str, help = 'folder name for saving fine tune model')

# Resnet Architecture
parser.add_argument('-depth', default=20, type=int, help='Model depth.')

# Random Erasing
parser.add_argument('-p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('-sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('-r1', default=0.3, type=float, help='aspect of erasing area')

# Training Setting
parser.add_argument('--only_fine_tune', default = 0, type = int, help = 'only fine tune')
parser.add_argument('--nepoch', default = 300, type = int, help = 'total number of training epochs')
parser.add_argument('--lr_decay_time', default = [150, 225], type = int, nargs= '+', help = 'when to multiply lr by 0.1')
parser.add_argument('--init_lr', default = 0.1, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
parser.add_argument('--batch_train', default = 128, type = int, help = 'batch size for training')
parser.add_argument('--batch_test', default = 128, type = int, help = 'batch size for testing')


# Fine Tuning Setting
parser.add_argument('--nepoch_fine_tune', default = 100, type = int, help = 'total number of training epochs in fine tuning')
parser.add_argument('--lr_decay_time_fine_tune', default = [], type = int, nargs= '*', help = 'when to multiply lr by 0.1 in fine tuning')
parser.add_argument('--init_lr_fine_tune', default = 0.001, type = float, help = 'initial learning rate in fine tuning')
parser.add_argument('--momentum_fine_tune', default = 0.9, type = float, help = 'momentum in SGD in fine tuning')

# Prior Setting
parser.add_argument('--sigma0', default = 0.000004, type = float, help = 'sigma_0^2 in prior')
parser.add_argument('--sigma1', default = 0.04, type = float, help = 'sigma_1^2 in prior')
parser.add_argument('--lambdan', default = 0.0000001, type = float, help = 'lambda_n in prior')
parser.add_argument('--prune_ratio', default = 0.1, type = float, help = 'prune ratio in prior')



args = parser.parse_args()




def model_eval(net, data_loader, device, loss_func):
    net.eval()
    correct = 0
    total_loss = 0
    total_count = 0
    for cnt, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = loss_func(outputs, labels)
        prediction = outputs.data.max(1)[1]
        correct += prediction.eq(labels.data).sum().item()
        total_loss += loss.mul(images.shape[0]).item()
        total_count += images.shape[0]

    return  1.0 * correct / total_count, total_loss / total_count

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize,
                                          transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3)])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         normalize])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    batch_size = args.batch_train

    np.random.seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_train, shuffle=True,num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_test, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_func = nn.CrossEntropyLoss().to(device)

    lambda_n = args.lambdan
    prior_sigma_0 = args.sigma0
    prior_sigma_1 = args.sigma1

    net = resnet_vb.ResNet_sparse_VB(args.depth, 10, lambda_n, prior_sigma_0, prior_sigma_1).to(device)


    optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=0)


    PATH = args.base_path + args.model_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    num_epochs = args.nepoch
    train_accuracy_path = np.zeros(num_epochs)
    train_loss_path = np.zeros(num_epochs)

    test_accuracy_path = np.zeros(num_epochs)
    test_loss_path = np.zeros(num_epochs)
    sparsity_path = np.zeros(num_epochs)

    torch.manual_seed(args.seed)

    NTrain = len(train_loader.dataset)
    best_accuracy = 0

    for epoch in range(num_epochs):
        net.train()
        epoch_training_loss = 0.0
        total_count = 0
        accuracy = 0

        if epoch in args.lr_decay_time:
            for para in optimizer.param_groups:
                para['lr'] = para['lr'] / 10

        if epoch < args.lr_decay_time[0]:
            prior_sigma_0 = args.sigma1
            net.set_prior(lambda_n, prior_sigma_0, prior_sigma_1)
        else:
            prior_sigma_0 = args.sigma0
            net.set_prior(lambda_n, prior_sigma_0, prior_sigma_1)
        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            output = net(input)
            loss = loss_func(output, target)

            optimizer.zero_grad()

            loss.backward()

            with torch.no_grad():
                for para in net.parameters():
                    prior_grad = para.prior_grad.div(NTrain)
                    para.grad.data +=  prior_grad

            optimizer.step()

            epoch_training_loss += loss.mul(input.shape[0]).item()
            accuracy += output.data.argmax(1).eq(target.data).sum().item()
            total_count += input.shape[0]
            train_loss_path[epoch] = epoch_training_loss / total_count
            train_accuracy_path[epoch] = accuracy / total_count
        print("epoch: ", epoch, ", train loss: ", epoch_training_loss / total_count, "train accuracy: ",
              accuracy / total_count)

        # calculate test set accuracy
        with torch.no_grad():
            print('sigma0:', net.sigma_0, 'sigma1:', net.sigma_1, 'lambdan:', net.lambda_n)

            test_accuracy, test_loss = model_eval(net, test_loader, device, loss_func)
            test_loss_path[epoch] = test_loss
            test_accuracy_path[epoch] = test_accuracy
            print("epoch: ", epoch, ", test loss: ", test_loss, "test accuracy: ", test_accuracy)


            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), PATH + 'best_model.pt')

            print('best accuracy:', best_accuracy)

        torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')

    import pickle
    filename = PATH + 'result.txt'
    f = open(filename, 'wb')
    pickle.dump([train_loss_path, train_accuracy_path, test_loss_path, test_accuracy_path, sparsity_path], f)
    f.close()

    #-----------------fine tune-------------_#
    PATH = args.base_path + args.model_path
    net.load_state_dict(torch.load(PATH + 'model' + str(args.nepoch - 1) + '.pt'))
    test_accuracy, test_loss = model_eval(net, test_loader, device, loss_func)
    print("test loss: ", test_loss, "test accuracy: ", test_accuracy)


    with torch.no_grad():
        total_num_para = 0
        epsilon = 1e-20
        for name, para in net.named_parameters():
            if 'mu' in name:
                total_num_para += para.numel()
        ratio_array = torch.zeros(total_num_para)
        count = 0
        for name, para in net.named_parameters():
            if 'mu' in name:
                temp_mu = para
            if 'rho' in name:
                temp_rho = para
                temp_sigma = torch.log(1 + torch.exp(para))
                temp_ratio = temp_mu.abs().div(temp_sigma + epsilon)
                size = para.numel()
                ratio_array[count:(count + size)] = temp_ratio.view(-1)
                count = count + size
        user_mask = {}
        target_ratio = args.prune_ratio
        threshold_index = int(np.floor(total_num_para * (1 - target_ratio)))
        temp = ratio_array.sort().values
        ratio_threshold = temp[threshold_index]
        count = 0
        for name, para in net.named_parameters():
            if 'mu' in name:
                size = para.numel()
                mask = (ratio_array[count:(count + size)] < ratio_threshold).view(para.shape)
                user_mask[name] = mask
                count = count + size
            if 'rho' in name:
                user_mask[name] = mask
    net.set_prune(user_mask)
    test_accuracy, test_loss = model_eval(net, test_loader, device, loss_func)
    print("test loss: ", test_loss, "test accuracy: ", test_accuracy)


    total_num_para = 0
    non_zero_element = 0
    for name, para in net.named_parameters():
        if 'mu' in name:
            total_num_para += para.numel()
            non_zero_element += (para != 0).sum()
    print('sparsity:', non_zero_element.item() / total_num_para)

    PATH = args.base_path + args.model_path + args.fine_tune_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr_fine_tune, momentum=args.momentum_fine_tune, weight_decay=5e-4)

    num_epochs = args.nepoch_fine_tune
    train_accuracy_path_fine_tune = np.zeros(num_epochs)
    train_loss_path_fine_tune = np.zeros(num_epochs)

    test_accuracy_path_fine_tune = np.zeros(num_epochs)
    test_loss_path_fine_tune = np.zeros(num_epochs)

    sparsity_path_fine_tune = np.zeros(num_epochs)

    torch.manual_seed(args.seed)

    best_accuracy = 0

    prior_sigma_0 = args.sigma1
    net.set_prior(lambda_n, prior_sigma_0, prior_sigma_1)


    for epoch in range(num_epochs):
        net.train()
        epoch_training_loss = 0.0
        total_count = 0
        accuracy = 0

        if epoch in args.lr_decay_time_fine_tune:
            for para in optimizer.param_groups:
                para['lr'] = para['lr']/10
        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            output = net(input)
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            epoch_training_loss += loss.mul(input.shape[0]).item()
            accuracy += output.data.argmax(1).eq(target.data).sum().item()
            total_count += input.shape[0]
            train_loss_path_fine_tune[epoch] = epoch_training_loss / total_count
            train_accuracy_path_fine_tune[epoch] = accuracy / total_count
        print("epoch: ", epoch, ", train loss: ", epoch_training_loss / total_count, "train accuracy: ",
              accuracy / total_count)

        with torch.no_grad():

            test_accuracy, test_loss = model_eval(net, test_loader, device, loss_func)
            test_loss_path_fine_tune[epoch] = test_loss
            test_accuracy_path_fine_tune[epoch] = test_accuracy
            print("epoch: ", epoch, ", test loss: ", test_loss, "test accuracy: ", test_accuracy)

            total_num_para = 0
            non_zero_element = 0
            for name, para in net.named_parameters():
                if 'mu' in name:
                    total_num_para += para.numel()
                    non_zero_element += (para.abs() != 0).sum()
            print('sparsity:', non_zero_element.item() / total_num_para)
            sparsity_path_fine_tune[epoch] = non_zero_element.item() / total_num_para

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), PATH + 'best_model.pt')
            print('best accuracy:', best_accuracy)

        torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')

    import pickle
    filename = PATH + 'result.txt'
    f = open(filename, 'wb')
    pickle.dump([train_loss_path_fine_tune, train_accuracy_path_fine_tune, test_loss_path_fine_tune,
                 test_accuracy_path_fine_tune, sparsity_path_fine_tune], f)
    f.close()


if __name__ == '__main__':
    main()
