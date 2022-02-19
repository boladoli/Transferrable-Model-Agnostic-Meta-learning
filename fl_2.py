from ult import get_data, cnn, mlp, Average, calcRMSE, \
    calcMAE, calcMAPE, reverse
import torch
from torch import nn
import copy
import random
import os
torch.set_printoptions(threshold=10_000)
import matplotlib.pyplot as plt
import numpy as np
import time

def eval_data(users, start_day_val, end_day_val):
    val_data_x = []
    val_data_y = []
    for user in users:
        valid_x, valid_y = get_data(user, start_day_val, end_day_val)
        val_data_x.append(valid_x)
        val_data_y.append(valid_y)
    x = torch.cat(val_data_x, dim=0)
    y = torch.cat(val_data_y, dim=0)
    return x, y


def local_model_update_fine(user, localEpoch, model, batch_size,
                       loss_function, optimizer, global_parameters, start_day, end_day):
    X, Y = get_data(user, start_day, end_day)
    model.load_state_dict(global_parameters, strict=True)
    model.train()
    for _ in range (localEpoch):
        permutation = torch.randperm(X.size()[0])
        for i in range(0, X.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X[indices], Y[indices]
            preds = model(batch_x)
            loss = loss_function(preds, batch_y)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())


def eval(parameters, model, val_x, val_y):
    loss_func = nn.MSELoss()
    model.load_state_dict(parameters, strict=True)
    model = model.eval()
    with torch.no_grad():
        val_pred = model(val_x)
        val_loss = loss_func(val_y, val_pred).item()
    return val_loss


def FL(edges_users, start_t, end_t, start_v, end_v, epochs, batch_size, lr):
    #torch.manual_seed(10)
    #torch.cuda.manual_seed_all(3)
    start_day = start_t
    end_day = end_t
    start_day_val = start_v
    end_day_val = end_v
    #######################################
    print("the number of edge {} ".format(len(edges_users)))
    model = mlp().cuda()
    #model = cnn().cuda()
    ###################################
    Epochs = epochs
    localEpoch = 5


    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    cfraction = 0.1
    batch_size = batch_size
    ##################################################
    global_parameters = {}
    for key, var in model.state_dict().items():
        global_parameters[key] = var.clone()
    best_weights = copy.deepcopy(model.state_dict())
    outputs_dir = 'C:/Users/heyu1/PycharmProjects/meta_learning/models_3'
    #################################################
    val_x, val_y = eval_data(edges_users, start_day_val, end_day_val)
    ###################################################
    b_loss = 100

    num_in_comm = int(max(len(edges_users) * cfraction, 1))
    v_loss = []
    b_epoch = 0

    model = model.train()
    for i in range(Epochs):

        start_time = time.time()
        clients_in_comm = random.sample(edges_users, num_in_comm)
        sum_parameters = None
        for client in clients_in_comm:
            local_parameters = local_model_update_fine(client, localEpoch, model, batch_size,
                       loss_func, optimizer, global_parameters, start_day, end_day)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)


        val_loss = eval(global_parameters, model, val_x, val_y)
        print("communicate round {}, total of {}, loss {}".format(i + 1, Epochs, val_loss))
        v_loss.append(val_loss)
        if val_loss < b_loss:
            b_loss = val_loss
            b_epoch = i
            best_weights = copy.deepcopy(model.state_dict())

        #print("--- %s seconds ---" % (time.time() - start_time))
        #break

    print("best epoch {} and val loss {}".format(b_epoch, b_loss))

    #torch.save(best_weights, os.path.join(outputs_dir, 'fed_cnn_1_1.pth'))
    return v_loss


def compared_pecon(edge_users):
    url = 'C:/Users/heyu1/PycharmProjects/meta_learning/models_3/fed_cnn_1_1.pth'
    #model = mlp().cuda()
    model = cnn().cuda()

    model.load_state_dict(torch.load(url))
    model.eval()
    print(len(edge_users))
    MAPE = []
    RMSE = []
    MAE = []
    for user in edge_users:
        test_x, test_y = get_data(user, '2013-08-08', '2013-08-22')

        with torch.no_grad():
            pred = np.array(torch.flatten(model(test_x)).cpu())
            test_y = np.array(torch.flatten(test_y).cpu())

            pred = reverse(pred)
            test_y = reverse(test_y)

            """plt.figure(figsize=(12, 6))
            plt.ylabel('Power [kW]', fontsize=25)
            plt.xlabel('Time index', fontsize=25)
            plt.plot(pred, linestyle='-.', label='pred', linewidth=2)
            plt.plot(test_y, 'm', linestyle='--', label='test_y', linewidth=2)
            plt.xticks(size=18)
            plt.yticks(size=18)
            plt.legend(loc='upper right', fontsize=19)
            plt.show()"""

            mape = calcMAPE(test_y, pred)
            rmse = calcRMSE(test_y, pred)
            mae = calcMAE(test_y, pred)

            print("user: {} mape: {}, rmse: {}, mae: {}".format(user, mape, rmse, mae))
            MAPE.append(mape.item())
            RMSE.append(rmse.item())
            MAE.append(mae.item())
    print(MAPE, RMSE, MAE)
    print("avg: ", Average(MAPE), Average(RMSE), Average(MAE))


if __name__ == "__main__":

    edges_users = ['user257', 'user259', 'user260', 'user262', 'user266', 'user268', 'user269', 'user271', 'user274',
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

    #start_t = '2013-08-04'
    #start_t = '2013-08-02'
    start_t = '2013-07-29'
    end_t = '2013-08-05'

    start_v = '2013-08-05'
    end_v = '2013-08-08'

    epochs = 500
    batch_size = 24
    # fl lr = 0.01, cnn lr = 0.01

    lr = 0.1

    #start_time = time.time()

    v_loss = FL(edges_users=edges_users, start_t=start_t, end_t=end_t, start_v=start_v,
                end_v=end_v, epochs=epochs, batch_size=batch_size, lr=lr)

    #print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure(figsize=(8, 5))
    plt.ylabel('Power[kW]', fontsize=12)
    plt.xlabel('Time index', fontsize=12)
    plt.plot(v_loss, label='fed_avg', linewidth=1)
    plt.legend(loc='best', fontsize=9)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.tight_layout()
    plt.show()

    compared_pecon(edge_users=edges_users)

    #871 845 835 829 507 504
