import torch
from anfis import Net


def _make_param(var):
    if isinstance(var, torch.Tensor):
        var = var.item()
    return torch.nn.Parameter(torch.tensor(var, dtype=torch.float))


class GaussianMF(torch.nn.Module):
    def __init__(self, mu, sigma):
        super(GaussianMF, self).__init__()
        self.register_parameter('mu', _make_param(mu))
        self.register_parameter('sigma', _make_param(sigma))

    def forward(self, x):
        var = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma ** 2))
        return var

    def print_params(self):
        return 'GaussianMF with {} and {}'.format(self.mu, self.sigma)


def gaussian_make(sigma, mus):
    return [GaussianMF(mu, sigma) for mu in mus]


class BellMF(torch.nn.Module):
    def __init__(self, a, b, c):
        super(BellMF, self).__init__()
        self.register_parameter('a', _make_param(a))
        self.register_parameter('b', _make_param(b))
        self.register_parameter('c', _make_param(c))

    @staticmethod
    def b_log(grad):
        grad[torch.isnan(grad)] = 1e-9

        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c) / self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))

    def print_params(self):
        return 'BellMF with {}, {} and {}'.format(self.a, self.b, self.c)


def bell_make(a, b, cs):
    return [BellMF(a, b, c) for c in cs]


def net_make():
    in_vars = [
            ('x', bell_make(2.5, 2, [1, 6])),
            ('y', bell_make(2.5, 2, [1, 6])),
            ('z', bell_make(2.5, 2, [1, 6])),
            ('k', bell_make(2.5, 2, [1, 6]))
            ]

    out_vars = ['output']
    model = Net('Simple classifier', in_vars, out_vars)
    return model

