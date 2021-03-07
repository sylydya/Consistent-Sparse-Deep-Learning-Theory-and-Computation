import numpy as np
import pandas as pd
import os
import errno

def my_relu(x):
    return x*(x>0)



#-----------------Regression Data------------------------_#
a = 1
b = 1

for my_seed in range(1,11):
    np.random.seed(my_seed)
    TotalP = 2000
    print('p = ', TotalP)
    NTrain = 10000
    x_train = np.matrix(np.zeros([NTrain, TotalP]))
    y_train = np.matrix(np.zeros([NTrain, 1]))

    sigma = 1.0
    for i in range(NTrain):
        if i%1000 == 0:
            print("x_train generate = ", i)
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        while ee > 10 or ee < -10:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            zj = np.sqrt(sigma) * np.random.normal(0, 1)
            while zj > 10 or zj < -10:
                zj = np.sqrt(sigma) * np.random.normal(0, 1)
            x_train[i, j] = (a*ee + b*zj) / np.sqrt(a*a+b*b)
        x0 = x_train[i, 0]
        x1 = x_train[i, 1]
        x2 = x_train[i, 2]
        x3 = x_train[i, 3]
        x4 = x_train[i, 4]

        y_train[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

    Nval = 1000
    x_val = np.matrix(np.zeros([Nval, TotalP]))
    y_val = np.matrix(np.zeros([Nval, 1]))

    sigma = 1.0
    for i in range(Nval):
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        while ee > 10 or ee < -10:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            zj = np.sqrt(sigma) * np.random.normal(0, 1)
            while zj > 10 or zj < -10:
                zj = np.sqrt(sigma) * np.random.normal(0, 1)
            x_val[i, j] = (a*ee + b*zj) / np.sqrt(a*a+b*b)
        x0 = x_val[i, 0]
        x1 = x_val[i, 1]
        x2 = x_val[i, 2]
        x3 = x_val[i, 3]
        x4 = x_val[i, 4]

        y_val[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

    NTest = 1000
    x_test = np.matrix(np.zeros([NTest, TotalP]))
    y_test = np.matrix(np.zeros([NTest, 1]))

    for i in range(NTest):
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        while ee > 10 or ee < -10:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            zj = np.sqrt(sigma) * np.random.normal(0, 1)
            while zj > 10 or zj < -10:
                zj = np.sqrt(sigma) * np.random.normal(0, 1)
            x_test[i, j] = (a*ee + b*zj) / np.sqrt(a*a+b*b)
        x0 = x_test[i, 0]
        x1 = x_test[i, 1]
        x2 = x_test[i, 2]
        x3 = x_test[i, 3]
        x4 = x_test[i, 4]

        y_test[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

    x_train_df = pd.DataFrame(x_train)
    y_train_df = pd.DataFrame(y_train)

    x_val_df = pd.DataFrame(x_val)
    y_val_df = pd.DataFrame(y_val)

    x_test_df = pd.DataFrame(x_test)
    y_test_df = pd.DataFrame(y_test)

    PATH = './data/regression/' + str(my_seed) + "/"
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    print("write train")

    x_train_df.to_csv(PATH + "x_train.csv")
    y_train_df.to_csv(PATH + "y_train.csv")
    print("write val")

    x_val_df.to_csv(PATH + "x_val.csv")
    y_val_df.to_csv(PATH + "y_val.csv")

    print('write test')

    x_test_df.to_csv(PATH + "x_test.csv")
    y_test_df.to_csv(PATH + "y_test.csv")



#--------------------Classification Data-----------------------------#

for my_seed in range(1,11):
    np.random.seed(my_seed)
    TotalP = 1000
    print('p = ', TotalP)
    NTrain = 10000
    x_train = np.matrix(np.zeros([NTrain, TotalP]))
    y_train = np.matrix(np.zeros([NTrain, 1]))
    sigma = 1
    current_positive = 0
    current_negtive = 0
    half_train = NTrain / 2

    for i in range(NTrain):


        if i%1000 == 0:
            print("x_train generate = ", i)
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        while ee > 10 or ee < -10:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            zj = np.sqrt(sigma) * np.random.normal(0, 1)
            while zj > 10 or zj < -10:
                zj = np.sqrt(sigma) * np.random.normal(0, 1)
            x_train[i, j] = (a*ee + b*zj) / np.sqrt(a*a+b*b)
        x0 = x_train[i, 0]
        x1 = x_train[i, 1]
        x2 = x_train[i, 2]
        x3 = x_train[i, 3]

        temp = np.exp(x0)+x1**2 + 5*np.sin(x2*x3)-3
        while current_positive >= half_train and temp > 0:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_train[i, j] = (a * ee + b * np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a * a + b * b)
            x0 = x_train[i, 0]
            x1 = x_train[i, 1]
            x2 = x_train[i, 2]
            x3 = x_train[i, 3]
            temp = np.exp(x0) + x1 ** 2 + 5 * np.sin(x2 * x3) - 3
        while current_negtive >= half_train and temp <= 0:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_train[i, j] = (a * ee + b * np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a * a + b * b)
            x0 = x_train[i, 0]
            x1 = x_train[i, 1]
            x2 = x_train[i, 2]
            x3 = x_train[i, 3]
            temp = np.exp(x0) + x1 ** 2 + 5 * np.sin(x2 * x3) - 3
        if temp > 0:
            y_train[i, 0] = 1
            current_positive += 1
        else:
            y_train[i,0] = 0
            current_negtive += 1

    Nval = 1000
    x_val = np.matrix(np.zeros([Nval, TotalP]))
    y_val = np.matrix(np.zeros([Nval, 1]))
    current_positive = 0
    current_negtive = 0
    half_val = Nval / 2

    for i in range(Nval):


        if i % 1000 == 0:
            print("x_val generate = ", i)
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        while ee > 10 or ee < -10:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            zj = np.sqrt(sigma) * np.random.normal(0, 1)
            while zj > 10 or zj < -10:
                zj = np.sqrt(sigma) * np.random.normal(0, 1)
            x_val[i, j] = (a * ee + b * zj) / np.sqrt(a * a + b * b)
        x0 = x_val[i, 0]
        x1 = x_val[i, 1]
        x2 = x_val[i, 2]
        x3 = x_val[i, 3]

        temp = np.exp(x0) + x1 ** 2 + 5 * np.sin(x2 * x3) - 3
        while current_positive >= half_val and temp > 0:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_val[i, j] = (a * ee + b * np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a * a + b * b)
            x0 = x_val[i, 0]
            x1 = x_val[i, 1]
            x2 = x_val[i, 2]
            x3 = x_val[i, 3]
            temp = np.exp(x0) + x1 ** 2 + 5 * np.sin(x2 * x3) - 3
        while current_negtive >= half_val and temp <= 0:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_val[i, j] = (a * ee + b * np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a * a + b * b)
            x0 = x_val[i, 0]
            x1 = x_val[i, 1]
            x2 = x_val[i, 2]
            x3 = x_val[i, 3]
            temp = np.exp(x0) + x1 ** 2 + 5 * np.sin(x2 * x3) - 3
        if temp > 0:
            y_val[i, 0] = 1
            current_positive += 1
        else:
            y_val[i, 0] = 0
            current_negtive += 1

    NTest = 1000
    x_test = np.matrix(np.zeros([NTest, TotalP]))
    y_test = np.matrix(np.zeros([NTest, 1]))

    current_positive = 0
    current_negtive = 0
    half_test = NTest / 2

    for i in range(NTest):


        if i % 1000 == 0:
            print("x_test generate = ", i)
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

        temp = np.exp(x0) + x1 ** 2 + 5 * np.sin(x2 * x3) - 3
        while current_positive >= half_test and temp > 0:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_test[i, j] = (a * ee + b * np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a * a + b * b)
            x0 = x_test[i, 0]
            x1 = x_test[i, 1]
            x2 = x_test[i, 2]
            x3 = x_test[i, 3]
            temp = np.exp(x0) + x1 ** 2 + 5 * np.sin(x2 * x3) - 3
        while current_negtive >= half_test and temp <= 0:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_test[i, j] = (a * ee + b * np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a * a + b * b)
            x0 = x_test[i, 0]
            x1 = x_test[i, 1]
            x2 = x_test[i, 2]
            x3 = x_test[i, 3]
            temp = np.exp(x0) + x1 ** 2 + 5 * np.sin(x2 * x3) - 3
        if temp > 0:
            y_test[i, 0] = 1
            current_positive += 1
        else:
            y_test[i, 0] = 0
            current_negtive += 1


    x_train_df = pd.DataFrame(x_train)
    y_train_df = pd.DataFrame(y_train)

    x_val_df = pd.DataFrame(x_val)
    y_val_df = pd.DataFrame(y_val)

    x_test_df = pd.DataFrame(x_test)
    y_test_df = pd.DataFrame(y_test)

    PATH = './data/classification/' + str(my_seed) + "/"
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    print("write train")

    x_train_df.to_csv(PATH + "x_train.csv")
    y_train_df.to_csv(PATH + "y_train.csv")
    print("write val")

    x_val_df.to_csv(PATH + "x_val.csv")
    y_val_df.to_csv(PATH + "y_val.csv")

    print('write test')

    x_test_df.to_csv(PATH + "x_test.csv")
    y_test_df.to_csv(PATH + "y_test.csv")




#--------------------Structure Selection Data-----------------------------#

for my_seed in range(1,11):
    np.random.seed(my_seed)
    TotalP = 1000
    print('p = ', TotalP)
    NTrain = 10000
    x_train = np.matrix(np.zeros([NTrain, TotalP]))
    y_train = np.matrix(np.zeros([NTrain, 1]))

    sigma = 0.5
    for i in range(NTrain):
        if i%1000 == 0:
            print("x_train generate = ", i)
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        while ee > 10 or ee < -10:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            zj = np.sqrt(sigma) * np.random.normal(0, 1)
            while zj > 10 or zj < -10:
                zj = np.sqrt(sigma) * np.random.normal(0, 1)
            x_train[i, j] = (a*ee + b*zj) / np.sqrt(a*a+b*b)
        x0 = x_train[i, 0]
        x1 = x_train[i, 1]
        x2 = x_train[i, 2]
        x3 = x_train[i, 3]
        x4 = x_train[i, 4]
        x5 = x_train[i, 5]
        x6 = x_train[i, 6]
        x7 = x_train[i, 7]
        x8 = x_train[i, 8]
        x9 = x_train[i, 9]

        y_train[i, 0] = np.tanh(2 * np.tanh(2 * x0 - x1)) + 2*np.tanh(
            2* np.tanh(x2 - 2 * x3) -  np.tanh(2*x4 )) + np.random.normal(0, 1)



    Nval = 1000
    x_val = np.matrix(np.zeros([Nval, TotalP]))
    y_val = np.matrix(np.zeros([Nval, 1]))

    sigma = 1.0
    for i in range(Nval):
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        while ee > 10 or ee < -10:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            zj = np.sqrt(sigma) * np.random.normal(0, 1)
            while zj > 10 or zj < -10:
                zj = np.sqrt(sigma) * np.random.normal(0, 1)
            x_val[i, j] = (a*ee + b*zj) / np.sqrt(a*a+b*b)
        x0 = x_val[i, 0]
        x1 = x_val[i, 1]
        x2 = x_val[i, 2]
        x3 = x_val[i, 3]
        x4 = x_val[i, 4]
        x5 = x_val[i, 5]
        x6 = x_val[i, 6]
        x7 = x_val[i, 7]
        x8 = x_val[i, 8]
        x9 = x_val[i, 9]

        y_val[i, 0] = np.tanh(2 * np.tanh(2 * x0 - x1)) + 2*np.tanh(
            2* np.tanh(x2 - 2 * x3) -  np.tanh(2*x4 )) + np.random.normal(0, 1)


    NTest = 1000
    x_test = np.matrix(np.zeros([NTest, TotalP]))
    y_test = np.matrix(np.zeros([NTest, 1]))

    for i in range(NTest):
        ee = np.sqrt(sigma) * np.random.normal(0, 1)
        while ee > 10 or ee < -10:
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
        for j in range(TotalP):
            zj = np.sqrt(sigma) * np.random.normal(0, 1)
            while zj > 10 or zj < -10:
                zj = np.sqrt(sigma) * np.random.normal(0, 1)
            x_test[i, j] = (a*ee + b*zj) / np.sqrt(a*a+b*b)
        x0 = x_test[i, 0]
        x1 = x_test[i, 1]
        x2 = x_test[i, 2]
        x3 = x_test[i, 3]
        x4 = x_test[i, 4]
        x5 = x_test[i, 5]
        x6 = x_test[i, 6]
        x7 = x_test[i, 7]
        x8 = x_test[i, 8]
        x9 = x_test[i, 9]

        y_test[i, 0] = np.tanh(2 * np.tanh(2 * x0 - x1)) + 2*np.tanh(
            2* np.tanh(x2 - 2 * x3) -  np.tanh(2*x4 )) + np.random.normal(0, 1)




    x_train_df = pd.DataFrame(x_train)
    y_train_df = pd.DataFrame(y_train)

    x_val_df = pd.DataFrame(x_val)
    y_val_df = pd.DataFrame(y_val)

    x_test_df = pd.DataFrame(x_test)
    y_test_df = pd.DataFrame(y_test)


    PATH = './data/structure/' + str(my_seed) + "/"
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    print("write train")

    x_train_df.to_csv(PATH +  "/x_train.csv")
    y_train_df.to_csv(PATH + "/y_train.csv")

    print("write val")

    x_val_df.to_csv(PATH + "/x_val.csv")
    y_val_df.to_csv(PATH + "/y_val.csv")

    print('write test')

    x_test_df.to_csv(PATH + "/x_test.csv")
    y_test_df.to_csv(PATH + "/y_test.csv")
