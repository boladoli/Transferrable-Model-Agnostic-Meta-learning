import torch
from torch import nn
from ult import Average, calcMAPE, calcRMSE, calcMAE, get_data, mlp, reverse, cnn
import copy
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from copy import deepcopy
import numpy as np


#####################################
def train_aggre(user_list, epochs, start_t, end_t, start_v, end_v, batch_size, lr):
    #model = mlp().cuda()
    model = cnn().cuda()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    outputs_dir = 'C:/Users/heyu1/PycharmProjects/meta_learning/models_3'
    validation_users = ['user172', 'user173', 'user174', 'user178', 'user183', 'user189', 'user190', 'user191',
                         'user193', 'user197']

    x = []
    x_v = []
    y = []
    y_v = []

    for user in user_list:
        train_x, train_y = get_data(user, start_t, end_t)
        x.append(train_x)
        y.append(train_y)
    train_x = torch.cat(x, dim=0)
    train_y = torch.cat(y, dim=0)

    for household in validation_users:
        val_x, val_y = get_data(household, start_v, end_v)
        x_v.append(val_x)
        y_v.append(val_y)
    val_x = torch.cat(x_v, dim=0)
    val_y = torch.cat(y_v, dim=0)


    b_loss = 1000
    train_x, train_y = Variable(train_x), Variable(train_y)
    val_x, val_y = Variable(val_x), Variable(val_y)
    model = model.train()

    for e in range(epochs):
        permutation = torch.randperm(train_x.size()[0])
        for i in range(0, train_x.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = train_x[indices], train_y[indices]
            preds = model(batch_x)
            loss = loss_func(preds, batch_y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_pred = model(val_x)
            val_loss = loss_func(y_pred, val_y).item()
            print(e, val_loss)
            if val_loss < b_loss:
                b_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
    torch.save(best_weights, os.path.join(outputs_dir, 'cnn_pretrain.pth'.format(user)))


def finetunning(users, tune_epochs, start_t, end_t,
                start_v, end_v, start_test, end_test, model_dir, out_dir, lr):

    outputs_dir = out_dir
    url = model_dir

    model = mlp().cuda()
    #model = cnn().cuda()
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


        if tune_epochs == 0:
            b_weights = deepcopy(model.state_dict())

        model.load_state_dict(b_weights, strict=True)
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


if __name__ == "__main__":


    source_users = [875, 876, 878, 881, 882, 833, 887, 888, 891, 898,
     901, 902, 903, 905, 908, 910, 911, 916, 917, 921,
     925, 927, 940, 935, 941, 948, 958, 965, 966, 968]
    source_users = ['user{}'.format(i) for i in source_users]


    target_users = ['user257', 'user259', 'user260', 'user262', 'user266', 'user268', 'user269', 'user271', 'user274',
                    'user276', 'user278', 'user280', 'user295', 'user298', 'user303', 'user468', 'user469', 'user471',
                    'user475',
                    'user478', 'user480', 'user498', 'user499',
                    'user500', 'user501', 'user502', 'user504', 'user505', 'user506', 'user507', 'user509', 'user515',
                    'user516', 'user549',
                    'user578', 'user594', 'user596', 'user829', 'user830', 'user832', 'user833', 'user835', 'user838',
                    'user842',
                    'user843', 'user845', 'user850',
                    'user852', 'user853', 'user854', 'user855', 'user857', 'user858', 'user860', 'user861', 'user865',
                    'user866',
                    'user869', 'user871', 'user873']
    print("source {}, target {}".format(len(source_users), len(target_users)))


    #train_aggre(user_list=source_users, epochs=450, start_t='2013-07-07', end_t='2013-07-29', start_v='2013-07-07',
    #            end_v='2013-07-29', batch_size=64, lr=0.1)

    finetunning(users=target_users, tune_epochs=0, start_t='2013-08-04', end_t='2013-08-05', start_v='2013-08-05', end_v='2013-08-08',
                start_test='2013-08-08', end_test='2013-08-22', model_dir='C:/Users/heyu1/PycharmProjects/meta_learning/models_3/fc_pretrain.pth',
                out_dir='C:/Users/heyu1/PycharmProjects/meta_learning/models_3/pre_train/fc/1 day', lr=0.1)


