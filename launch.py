import torch.nn.functional as F
from memberfuncs import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import itertools
import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

    return total_loss, rmse, percent_loss


def test_net(model, data, show_plots=True):
    x, y = data.dataset.tensors

    if show_plots:
        memberfunc_allvars_plot(model, x)

    print('Test model with {} cases'.format(x.shape[0]))
    preds = model(x)
    total_loss, rmse, percent_loss = error_find(preds, y)
    print('MSE {:.5f}, RMSE {:.5f}'.format(total_loss, rmse))

    if show_plots:
        results_plot(y, preds)


def train_net_process(model, data, optimizer, criterion, epochs, show_plot):
    errors = []
    print('Training net with {} epochs and {} train cases'.format(epochs, data.dataset.tensors[0].shape[0]))

    for epoch in range(epochs):
        # For each batch
        for x, y in data:
            preds = model(x)
            loss = criterion(y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        x, y = data.dataset.tensors

        with torch.no_grad():
            model.fit_coeffs(x, y)
        preds = model(x)
        total_loss, rmse, percent_loss = error_find(preds, y)
        errors.append(rmse)

        if epoch < 30 or epoch % 10 == 0:
            print('Epoch {:4d}. Errors: MSE {:.5f}, RMSE {:.5f}'.format(epoch, total_loss, rmse
                                                                        ))

    if show_plot:
        errors_plot(errors)
        y = data.dataset.tensors[1]
        preds = model(data.dataset.tensors[0])
        results_plot(y, preds)


def train_net(model, data, epochs, show_plot=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = torch.nn.MSELoss(reduction='sum')
    train_net_process(model, data, optimizer, criterion, epochs, show_plot)


##### Making Dataset #####

def ex2_eqn(x, y, z, k):
    output = 1 + torch.pow(x, 0.5) + torch.pow(y, -1) + torch.pow(z, -1.5) + torch.pow(k, -2.0)
    output = torch.pow(output, 2)
    return output


def _make_data_xyz():
    # xyz_vals = itertools.product(inp_range, inp_range, inp_range, inp_range)
    # x = torch.tensor(list(xyz_vals), dtype=dtype)
    # y = torch.tensor([[ex2_eqn(*p)] for p in x], dtype=dtype)
    data = pd.read_excel(r'C:\Users\maxim\OneDrive\Desktop\folder\diplom\data\parsing\companies_scores.xlsx')
    data = data[data['target'] != 0]
    x = torch.tensor(data[['product','tech','org']].values, dtype=dtype)
    y = torch.tensor(data['target'].values, dtype=dtype)

    return TensorDataset(x, y)


def train_create(batch_size=100):
    #inp_range = range(1, 7, 1)
    td = _make_data_xyz()

    return DataLoader(td, batch_size=batch_size, shuffle=True)


def test_create():
    inp_range = np.arange(1.5, 6.5, 1)
    td = _make_data_xyz(inp_range)
    return DataLoader(td)


if __name__ == '__main__':
    show_plots = True
    model = net_make()
    # train_data = train_create()
    # train_net(model, train_data, 500, show_plots)
    # test_data = test_create()
    # test_net(model, test_data, show_plots)
