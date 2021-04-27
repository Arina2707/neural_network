import torch.nn.functional as F
from memberfuncs import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import itertools
import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

dtype = torch.float


def errors_plot(errors):
    plt.plot(range(len(errors)), errors, '-ro', label='errors')
    plt.ylabel('RMSE error')
    plt.xlabel('Epoch')
    plt.show()


def results_plot(y_actual, preds):
    plt.plot(range(len(preds)), preds.detach().numpy(),
             'r', label='trained')
    plt.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='original')
    plt.legend(loc='upper left')
    plt.show()


def _memberfunc_plot(var, funval, x):
    x_sorted, _ = x.sort()

    for mf_name, yvals in funval.fuzzify(x_sorted):
        plt.plot(x_sorted.tolist(), yvals.tolist(), label=mf_name)
    plt.xlabel('Variable {} with {} membership functions'.format(var, funval.mfs_num))
    plt.ylabel('Membership function value')
    plt.legend(bbox_to_anchor=(1., 0.95))
    plt.show()


def memberfunc_allvars_plot(model, x):
    for i, (var, funval) in enumerate(model.layer.fuzzify.vars.items()):
        _memberfunc_plot(var, funval, x[:, i])


def error_find(preds, y_actual):
    with torch.no_grad():
        total_loss = F.mse_loss(preds, y_actual)
        rmse = torch.sqrt(total_loss).item()
        percent_loss = torch.mean(100. * torch.abs((preds - y_actual) / y_actual))

    return rmse, percent_loss


def test_net(model, data, show_plots=False):
    x, y = data.dataset.tensors
    #
    # if show_plots:
    #     memberfunc_allvars_plot(model, x)
    #
    # print('Test model with {} cases'.format(x.shape[0]))
    preds = model(x)
    # total_loss, rmse, percent_loss = error_find(preds, y)
    # print('MSE {:.5f}, RMSE {:.5f}'.format(total_loss, rmse))
    print(preds)
    if show_plots:
        results_plot(y, preds)


def train_net_process(model, data, optimizer, criterion, epochs, show_plot):
    errors = []
    print('Training net with {} epochs and {} train cases'.format(epochs, data.dataset.tensors[0].shape[0]))

    for epoch in range(epochs):
        print('Epoch: ', epoch)

        for x, y in data:
            preds = model(x)
            loss = criterion(y.unsqueeze(1), preds)
            #errors.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x, y = data.dataset.tensors
        with torch.no_grad():
             model.fit_coeffs(x, y)
        preds = model(x)
        rmse, percent_loss = error_find(preds, y.unsqueeze(1))
        errors.append(rmse)

        # if epoch < 30 or epoch % 10 == 0:
        #     print('Epoch {:4d}. Errors: RMSE {:.5f}'.format(epoch, rmse))
        # print(model.coeffs)


    if show_plot:
        errors_plot(errors)
        y = data.dataset.tensors[1]
        preds = model(data.dataset.tensors[0])
        results_plot(y, preds)


def train_net(model, data, epochs, show_plot=True):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = torch.nn.MSELoss(reduction='sum')
    train_net_process(model, data, optimizer, criterion, epochs, show_plot)


##### Making Dataset #####

def ex2_eqn(x, y, z, k):
    output = 1 + torch.pow(x, 0.5) + torch.pow(y, -1) + torch.pow(z, -1.5) + torch.pow(k, -2.0)
    output = torch.pow(output, 2)
    return output


def make_data_xyz():
    # data = pd.read_excel(r'C:\Users\maxim\OneDrive\Desktop\folder\diplom\data\parsing\companies_scores.xlsx')
    # data = data[data['target'] != 0]
    # data['product'] = data['product']*100
    # data['tech'] = data['tech'] * 100
    # data['org'] = data['org'] * 100
    # data['target'] = data['target'] * 100
    #
    # x = torch.tensor(data[['product','tech','org']].values, dtype=dtype)
    # y = torch.tensor(data['target'].values, dtype=dtype)

    ts = np.loadtxt("train_1.txt", usecols=[0,1,2,3,4])

    X = ts[:, 0:4]
    Y = ts[:, 4]

    # ts = np.loadtxt("trainingSet_notmy.txt", usecols=[1,2,3])
    #
    # X = ts[:, 0:2]
    # Y = ts[:, 2]

    x = torch.tensor(X, dtype=dtype)
    y = torch.tensor(Y, dtype=dtype)

    return TensorDataset(x, y)


def train_create(batch_size=9):
    td = make_data_xyz()

    return DataLoader(td, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    show_plots = True
    model = net_make()
    train_data = train_create()

    train_net(model, train_data, 25, show_plots)
    #test_net(model, train_data)
