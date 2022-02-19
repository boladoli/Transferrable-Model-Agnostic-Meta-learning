import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import higher
import random
from ult import get_data, mlp, cnn
from copy import deepcopy
import os
from pre_train3 import finetunning
import copy
import time

def select_points(user, k):
    X, Y = get_data(user, start_day='2013-07-07', end_day='2013-07-29')
    permutation = torch.randperm(X.size()[0])
    indices = permutation[0:0 + k]
    x, y = X[indices], Y[indices]
    return x, y


def evaluate(net, valx, valy):
    with torch.no_grad():
        loss_func = nn.MSELoss()
        out = net(valx)
        v_loss = loss_func(out, valy)
    return v_loss.item()


def predict_test(test_weights, inner_step_size, steps):
    validataion_users = ['user172', 'user173', 'user174', 'user178', 'user183', 'user189', 'user190', 'user191',
                         'user193', 'user197']
    vali_loss = 0.0
    #tmp_m = mlp().cuda()

    tmp_m = cnn().cuda()

    for user in validataion_users:
        tmp_m.load_state_dict(test_weights, strict=True)
        test_opt = torch.optim.SGD(tmp_m.parameters(), lr=inner_step_size)
        in_, target = select_points(user, 32)
        testx, testy = select_points(user, 32)
        loss_func = nn.MSELoss()
        for i in range(steps):
            tmp_m.zero_grad()
            out = tmp_m(in_)
            loss = loss_func(out, target)
            loss.backward()
            test_opt.step()
        v_loss = evaluate(tmp_m, testx, testy)
        vali_loss += v_loss
    vali_loss = vali_loss/len(validataion_users)
    return vali_loss


def train(tasks, net, meta_opt, K, task_num, inner_step_size):
    net.train()
    meta_opt.zero_grad()
    inner_opt = torch.optim.SGD(net.parameters(), lr=inner_step_size)
    n_inner_iter = 5

    for i in range(task_num):
        with higher.innerloop_ctx(net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            task = random.sample(tasks, 1)[0]
            for _ in range(n_inner_iter):
                x_spt, y_spt = select_points(task, K)
                spt_logits = fnet(x_spt)
                spt_loss = F.mse_loss(spt_logits, y_spt)
                diffopt.step(spt_loss)
            task = random.sample(tasks, 1)[0]
            x_qry, y_qry = select_points(task, K)
            qry_logits = fnet(x_qry)
            qry_loss = F.mse_loss(qry_logits, y_qry)
            qry_loss.backward()
    meta_opt.step()


def validate(test_weights):


    tmp_m = mlp().cuda()
    loss_func = nn.MSELoss()
    validataion_users = ['user172', 'user173', 'user174', 'user178', 'user183', 'user189', 'user190', 'user191',
                         'user193', 'user197']

    tmp_m.load_state_dict(test_weights, strict=True)
    y_v = []
    x_v = []

    for household in validataion_users:
        val_x, val_y = get_data(household, '2013-07-07', '2013-07-29')
        x_v.append(val_x)
        y_v.append(val_y)
    val_x = torch.cat(x_v, dim=0)
    val_y = torch.cat(y_v, dim=0)

    with torch.no_grad():
        y_pred = tmp_m(val_x)
        val_loss = loss_func(y_pred, val_y).item()
    return val_loss


def main():
    source_users = [875, 876, 878, 881, 882, 833, 887, 888, 891, 898,
                    901, 902, 903, 905, 908, 910, 911, 916, 917, 921,
                    925, 927, 940, 935, 941, 948, 958, 965, 966, 968]
    source_users = ['user{}'.format(i) for i in source_users]


    #net = mlp().cuda()
    net = cnn().cuda()
    meta_opt = optim.Adam(net.parameters(), lr=0.001)
    inner_step_size = 0.1
    fine_tune_step = 5

    best_vali = 1000
    best_epoch = 0

    for epoch in range(350):

        #start_time = time.time()
        train(source_users, net, meta_opt, K=12, task_num=20, inner_step_size=inner_step_size)
        test_weights = deepcopy(net.state_dict())
        val_l = predict_test(test_weights, 0.01, steps=fine_tune_step)
        #val_l = validate(test_weights)
        if val_l < best_vali:
            best_vali = val_l
            best_epoch = epoch
            best_weights = deepcopy(test_weights)
        print("epoch {} val_loss {}".format(epoch, val_l))

        #print("--- %s seconds ---" % (time.time() - start_time))
        #break

    print("best_epoch {}, best_val {}".format(best_epoch, best_vali))
    outputs_dir = 'C:/Users/heyu1/PycharmProjects/meta_learning/models_3'
    torch.save(best_weights, os.path.join(outputs_dir, 'cnn_meta_2.pth'))


if __name__ == "__main__":

    #main()
    target_users = ['user257', 'user259', 'user260', 'user262', 'user266', 'user268', 'user269', 'user271', 'user274',
                    'user276', 'user278', 'user280', 'user295', 'user298', 'user303', 'user468', 'user469', 'user471',
                    'user475', 'user478', 'user480', 'user498', 'user499',
                    'user500', 'user501', 'user502', 'user504', 'user505', 'user506', 'user507', 'user509', 'user515', 'user516', 'user549',
                    'user578', 'user594', 'user596', 'user829', 'user830', 'user832', 'user833', 'user835', 'user838',
                    'user842', 'user843', 'user845', 'user850',
                    'user852', 'user853', 'user854', 'user855', 'user857', 'user858', 'user860', 'user861', 'user865',
                    'user866', 'user869', 'user871', 'user873']

    print(len(target_users))

    finetunning(users=target_users, tune_epochs=5, start_t='2013-08-02', end_t='2013-08-05', start_v='2013-08-05',
                end_v='2013-08-08', start_test='2013-08-08', end_test='2013-08-22',
                model_dir='C:/Users/heyu1/PycharmProjects/meta_learning/models_3/fc_meta.pth',
                out_dir='C:/Users/heyu1/PycharmProjects/meta_learning/models_3/meta/cnn_diff_steps/7 day/50 steps', lr=0.1)

