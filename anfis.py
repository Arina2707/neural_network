import torch
import torch.nn.functional as F
import numpy as np
import itertools
from collections import OrderedDict

dtype = torch.float


class FuzzyVariable(torch.nn.Module):
    def __init__(self, mfs):
        super(FuzzyVariable, self).__init__()

        if isinstance(mfs, list):
            mfs_names = ['mfs{}'.format(i) for i in range(len(mfs))]
            mfs = OrderedDict(zip(mfs_names, mfs))

        self.mfs = torch.nn.ModuleDict(mfs)
        self.padding = 0

    @property
    def mfs_num(self):
        return len(self.mfs)

    def members(self):
        return self.mfs.items()

    def make_padding(self, new_size):
        self.padding = new_size - len(self.mfs)

    def fuzzify(self, x):
        # Fuzzy values for input variables
        for name, mf in self.mfs.items():
            val = mf(x)
            yield name, val

    def forward(self, x):
        preds = torch.cat([func(x) for func in self.mfs.values()], dim=1)

        if self.padding > 0:
            preds = torch.cat([preds, torch.zeros(x.shape[0], self.padding)], dim=1)

        return preds


class FuzzyLayer(torch.nn.Module):
    def __init__(self, vars, vars_names=None):
        super(FuzzyLayer, self).__init__()

        if not vars_names:
            self.vars_names = ['x{}'.format(i) for i in range(len(vars_names))]
        else:
            self.vars_names = list(vars_names)

        self.max_mfsnum = max([var.mfs_num for var in vars])

        for var in vars:
            var.make_padding(self.max_mfsnum)

        self.vars = torch.nn.ModuleDict(zip(self.vars_names, vars))

    @property
    def vars_num(self):
        return len(self.vars)

    @property
    def max_mfs(self):
        return self.max_mfsnum

    def forward(self, x):
        assert x.shape[1] == self.vars_num, '{} is wrong number of input params'.format(self.vars_num)
        # concatenates on new dimension
        preds = torch.stack([var(x[:, i:i + 1]) for i, var in enumerate(self.vars.values())], dim=1)

        return preds


class RulesLayer(torch.nn.Module):
    def __init__(self, vars):
        super(RulesLayer, self).__init__()
        mf_num = [var.mfs_num for var in vars]
        mf_idx = itertools.product(*[range(n) for n in mf_num])
        self.mf_idx = torch.tensor(list(mf_idx))

    def rules_num(self):
        return len(self.mf_idx)

    def forward(self, x):
        batch_idx = self.mf_idx.expand((x.shape[0], -1, -1))
        antecedents = torch.gather(x.transpose(1, 2), 1, batch_idx)
        rules = torch.prod(antecedents, dim=2)

        return rules


class ConsequentLayer(torch.nn.Module):
    def __init__(self, in_num, rule_num, out_num):
        super(ConsequentLayer, self).__init__()
        shape = torch.Size([rule_num, out_num, in_num + 1])
        self._coeffs = torch.zeros(shape, dtype=dtype, requires_grad=True)

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, new_coeffs):
        assert new_coeffs.shape == self.coeffs.shape, 'Coeffs shape is {}, but should be {}'.format(new_coeffs.shape,
                                                                                                    self.coeffs.shape)
        self._coeffs = new_coeffs

    def fit_coeffs(self, x, weights, y_actual):
        x_new = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_new)
        weighted_x[weighted_x == 0] = 1e-12
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)

        try:
            # Built-in torch function to solve the least squares problem of minimizing over coeffs
            coeffs_2d, _ = torch.lstsq(y_actual_2d, weighted_x_2d)
        except RuntimeError as err:
            print('Internal error in gels', err)
            print('Weights are', weighted_x)
            raise err

        coeffs_2d = coeffs_2d[0:weighted_x_2d.shape[1]]
        # coeffs dim is: n_rules * n_out * (n_in+1)
        self.coeffs = coeffs_2d.view(weights.shape[1], x.shape[1] + 1, -1).transpose(1, 2)

    def forward(self, x):
        '''
            Calculate: y = coeff * x + const   [NB: no weights yet]
            x.shape: n_cases * n_in
            coeff.shape: n_rules * n_out * (n_in+1)
            y.shape: n_cases * n_out * n_rules
        '''
        # Append one instead of constant
        x_new = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        preds = torch.matmul(self.coeffs, x_new.t().float())

        return preds.transpose(0, 2)


class PlainConsequentLayer(ConsequentLayer):
    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients', torch.nn.Parameter(self._coeffs))

    @property
    def coeffs(self):
        return self.coefficients

    def fit_coeffs(self, x, weights, y_actual):
        assert False, 'I am using Backpropagation instead of hybrid algorithm.'


class WeightedSumLayer(torch.nn.Module):
    def __init__(self):
        super(WeightedSumLayer, self).__init__()

    def forward(self, weights, tsk):
        # Performs a batch matrix-matrix product of matrices stored in x, y
        preds = torch.bmm(tsk, weights.unsqueeze(2))

        return preds.squeeze(2)


class Net(torch.nn.Module):
    def __init__(self, description, in_varsnames, out_varsnames, hybrid=True):
        super(Net, self).__init__()
        self.description = description
        self.out_varnames = out_varsnames
        self.hybrid = hybrid
        mfs_names = [var for var, _ in in_varsnames]
        mfs = [FuzzyVariable(mfs) for _, mfs in in_varsnames]
        self.in_num = len(in_varsnames)
        self.rules_num = np.prod([len(mfs) for _, mfs in in_varsnames])

        if self.hybrid:
            cl = ConsequentLayer(self.in_num, self.rules_num, self.out_num)
        else:
            cl = PlainConsequentLayer(self.in_num, self.rules_num, self.out_num)

        self.layer = torch.nn.ModuleDict(OrderedDict([('fuzzify', FuzzyLayer(mfs, mfs_names)), ('rules', RulesLayer(mfs)), ('consequent', cl)]))

    @property
    def out_num(self):
        return len(self.out_varnames)

    @property
    def coeffs(self):
        return self.layer['consequent'].coeffs

    @coeffs.setter
    def coeffs(self, new_coeffs):
        self.layer['consequent'].coeffs = new_coeffs

    def fit_coeffs(self, x, y_actual):
        if self.hybrid:
            self(x)
            self.layer['consequent'].fit_coeffs(x, self.weights, y_actual)

    def inputs_iterator(self):
        return self.layer['fuzzify'].vars.items()

    def outputs_iterator(self):
        return self.out_varnames

    def forward(self, x):
        self.fuzzified = self.layer['fuzzify'](x)
        self.initial_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.initial_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        preds = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2).float())
        self.preds = preds.squeeze(2)

        return self.preds
