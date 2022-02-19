import pandas as pd
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def Average(lst):
    return sum(lst) / len(lst)


def create_dataset(data, days_for_train) -> (np.array, np.array):
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return (np.array(dataset_x), np.array(dataset_y))


def getListOfFiles(dirName):

 listOfFile = os.listdir(dirName)
 allFiles = list()
 for entry in listOfFile:
  fullPath = os.path.join(dirName, entry)
  if os.path.isdir(fullPath):
   allFiles = allFiles + getListOfFiles(fullPath)
  else:
   allFiles.append(fullPath)
 return allFiles


# 计算RMSE
def calcRMSE(true,pred):
    return np.sqrt(mean_squared_error(true, pred))


# 计算MAE
def calcMAE(true,pred):
    return mean_absolute_error(true, pred)


# 计算MAPE
def calcMAPE(true, pred, epsion = 0.0000000):
    true += epsion
    return np.mean(np.abs((true-pred)/true))*100

def calcSMAPE(true, pred):
    delim = (np.abs(true)+np.abs(pred))/2.0
    return np.mean(np.abs((true-pred)/delim))*100

def get_data(user, start_day, end_day):
    full_path = os.path.join('C:/Users/heyu1/PycharmProjects/FL/data_full/sepreated_users', user)
    data_list = getListOfFiles(full_path)
    df = pd.read_csv(data_list[0])
    df['READING_DATETIME'] = pd.to_datetime(df['READING_DATETIME'])
    mask = (df['READING_DATETIME'] >= start_day) & (df['READING_DATETIME'] < end_day)
    select_time = df.loc[mask]
    power = np.array(select_time.iloc[:, 6])
    power = (power - 0) / (4 - 0)
    x, y = create_dataset(power, 12)

    #full connected
    x = torch.from_numpy(x.reshape(-1, 12)).float().cuda()

    #cnn
    #x = torch.from_numpy(x.reshape(-1, 1, 12)).float().cuda()
    y = torch.from_numpy(y.reshape(-1, 1)).float().cuda()
    return x, y


def reverse(data, max=4, min=0):
    data = data * (max - min) + min
    return data


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.first = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels=1, out_channels=15,
                                kernel_size=7)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv1d(in_channels=15, out_channels=15,
                                kernel_size=5)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

        self.linear=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(30, 30)),
            ('relu2', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(30, 1))
        ]))

    def forward(self, x):
        x = self.first(x)
        s, b, h = x.shape
        x = x.view(s, b * h)
        x = self.linear(x)
        return x


class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(12, 128)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(128, 128)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(128, 1))
        ]))
    def forward(self, x):
        return self.model(x)
###################
import time


def finetunning(users, tune_epochs, start_t, end_t, start_test, end_test, model_dir, out_dir, lr):

    outputs_dir = out_dir
    url = model_dir

    #model = mlp().cuda()
    model = cnn().cuda()
    model.load_state_dict(torch.load(url))
    global_weights = deepcopy(model.state_dict())

    mape_l = []
    rmse_l = []
    mae_l = []

    for user in users:



        train_x, train_y = get_data(user, start_t, end_t)
        test_x, test_y = get_data(user, start_test, end_test)

        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model.load_state_dict(global_weights, strict=True)

        # before the fine_tuning

        with torch.no_grad():
            pred_before = reverse(np.array(torch.flatten(model(test_x)).cpu()))

        #start_time = time.time()
        model.train()
        for step in range(tune_epochs):
            out = model(train_x)
            loss = loss_func(out, train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print("--- %s seconds ---" % (time.time() - start_time))
        #break
        with torch.no_grad():

            pred = np.array(torch.flatten(model(test_x)).cpu())
            test_y = np.array(torch.flatten(test_y).cpu())

            pred = reverse(pred)
            test_y = reverse(test_y)

            """plt.figure(figsize=(12, 6))
            plt.ylabel('Power [kW]', fontsize=25)
            plt.xlabel('Time index', fontsize=25)
            plt.plot(pred_before, linestyle='-.', label='before_fine', linewidth=2)
            plt.plot(pred, linestyle='-.', label='pred', linewidth=2)
            plt.plot(test_y, 'm', linestyle='--', label='test_y', linewidth=2)
            plt.xticks(size=18)
            plt.yticks(size=18)
            plt.legend(loc='upper right', fontsize=19)
            plt.show()"""


            mape = calcMAPE(test_y, pred)
            rmse = calcRMSE(test_y, pred)
            mae = calcMAE(test_y, pred)

            b_weights = deepcopy(model.state_dict())
            print(" {} mape: {}, rmse: {}, mae: {}".format(user, mape, rmse, mae))
            torch.save(b_weights, os.path.join(outputs_dir, '{}.pth'.format(user)))

            #plt.savefig('C:/Users/heyu1/PycharmProjects/meta_learning/imgs/5/{}'.format(user))

            mape_l.append(mape.item())
            rmse_l.append(rmse.item())
            mae_l.append(mae.item())

    print("average:")
    print("mape: ", mape_l)
    print("rmse: ", rmse_l)
    print("mae: ", mae_l)

    print(Average(mape_l), Average(rmse_l), Average(mae_l))



"""plt.figure(figsize=(12, 6))
plt.ylabel('Power [kW]', fontsize=25)
plt.xlabel('Time index', fontsize=25)
plt.plot(pred, linestyle='-.', label='pred', linewidth=2)
plt.plot(test_y, 'm', linestyle='--', label='test_y', linewidth=2)
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(loc='upper right', fontsize=19)
plt.show()
"""

"""plt.figure(figsize=(12, 6))
plt.ylabel('Power [kW]', fontsize=25)
plt.xlabel('Time index', fontsize=25)
plt.plot(pred_before, linestyle='-.', label='before_fine', linewidth=2)
plt.plot(pred, linestyle='-.', label='pred', linewidth=2)
plt.plot(test_y, 'm', linestyle='--', label='test_y', linewidth=2)
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(loc='upper right', fontsize=19)
plt.show()"""


#######################
"""
def finetunning(users, tune_epochs, start_t, end_t,
                start_v, end_v, start_test, end_test, model_dir, out_dir, lr):

    outputs_dir = out_dir
    url = model_dir

    model = mlp().cuda()
    model.load_state_dict(torch.load(url))
    global_weights = deepcopy(model.state_dict())

    mape_l = []
    rmse_l = []
    mae_l = []

    for user in users:

        train_x, train_y = get_data(user, start_t, end_t)
        valid_x, valid_y = get_data(user, start_v, end_v)
        test_x, test_y = get_data(user, start_test, end_test)

        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        model.load_state_dict(global_weights, strict=True)

        b_loss = 10000


        # before the fine_tuning
        with torch.no_grad():
            pred_before = reverse(np.array(torch.flatten(model(test_x)).cpu()))

        model.train()
        for step in range(tune_epochs):
            out = model(train_x)
            loss = loss_func(out, train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_pred = model(valid_x)
                val_loss = loss_func(y_pred, valid_y).item()

                if val_loss < b_loss:
                    b_loss = val_loss
                    b_weights = deepcopy(model.state_dict())

        model.load_state_dict(b_weights, strict=True)
        with torch.no_grad():

            pred = np.array(torch.flatten(model(test_x)).cpu())
            test_y = np.array(torch.flatten(test_y).cpu())

            pred = reverse(pred)
            test_y = reverse(test_y)


            mape = calcMAPE(test_y, pred)
            rmse = calcRMSE(test_y, pred)
            mae = calcMAE(test_y, pred)

            print(" {} mape: {}, rmse: {}, mae: {}".format(user, mape, rmse, mae))
            #torch.save(b_weights, os.path.join(outputs_dir, '{}.pth'.format(user)))

            mape_l.append(mape.item())
            rmse_l.append(rmse.item())
            mae_l.append(mae.item())

    print("average:")
    print("mape: ", mape_l)
    print("rmse: ", rmse_l)
    print("mae: ", mae_l)

    print(Average(mape_l), Average(rmse_l), Average(mae_l))

"""
