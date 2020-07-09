import math

import torch
import torch.nn as nn
import torch.autograd as autograd

from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.common.common import ndarray


class Controller(nn.Module):
    def __init__(self, deformable):
        super(Controller, self).__init__()
        self.deformable = deformable
        self.dofs = deformable.dofs()
        self.act_dofs = deformable.act_dofs()
        self.all_muscles = deformable.all_muscles

    def update_weight(self, weight):
        raise NotImplementedError

    def get_grad(self):
        raise NotImplementedError


class NNController(Controller):
    def __init__(self, deformable, widths, layer_norm=False, dropout=0.0):
        super(NNController, self).__init__(deformable)
        self.layers = nn.ModuleList()
        for i in range(len(widths) - 1):
            if i < len(widths) - 2:
                self.layers.append(nn.Linear(widths[i], widths[i + 1], bias=not layer_norm))
                if layer_norm:
                    self.layers.append(nn.LayerNorm(widths[i + 1], elementwise_affine=True))
                self.layers.append(nn.Tanh())
            else:
                if dropout > 0.0:
                    self.layers.append(nn.Dropout(p=dropout))
                self.layers.append(nn.Linear(widths[i], widths[i + 1], bias=True))

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, prev_a):
        raise NotImplementedError


class IndNNController(NNController):
    def __init__(self, deformable, widths, layer_norm=False, dropout=0.0):
        super(IndNNController, self).__init__(deformable, widths, layer_norm, dropout)
        self.layers[-1] = nn.Linear(self.layers[-1].in_features, self.act_dofs, bias=True)

    def forward(self, x, prev_a) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x)
        x = x.sigmoid()
        return x.squeeze(0)


class AdaNNController(NNController):
    def __init__(self, deformable, widths, layer_norm=False, dropout=0.0):
        super(AdaNNController, self).__init__(deformable, widths, layer_norm, dropout)
        self.a_init = torch.zeros(self.act_dofs, requires_grad=False)

    def forward(self, x, prev_a) -> torch.Tensor:
        if prev_a is None:
            prev_a = self.a_init
        a = []
        pointer = 0

        for layer in self.layers:
            x = layer(x)
        x = x.tanh().squeeze(0)

        for w, muscle_pair in zip(x, self.all_muscles):
            if len(muscle_pair) != 2:
                raise ValueError('adaptive controller require paired muscles')
            w = w.view(1)
            mu_pair = [0.5 * (w.abs() - w), 0.5 * (w.abs() + w)]

            for mu, muscle in zip(mu_pair, muscle_pair):
                prev_a_cord = prev_a[pointer:pointer + len(muscle)]
                pointer += len(muscle)
                a_cord = torch.cat([mu, prev_a_cord[:-1]])
                a.append(a_cord)
        a = torch.cat(a)
        return 1 - a


class SnakeAdaNNController(AdaNNController):
    def __init__(self, deformable, widths, layer_norm=False, dropout=0.0):
        super(SnakeAdaNNController, self).__init__(deformable, widths, layer_norm, dropout)
        self.prev_a_clean = None

    def forward(self, x, prev_a) -> torch.Tensor:
        if prev_a is None:
            self.prev_a_clean = self.a_init
        a = []
        a_clean = []
        pointer = 0

        for layer in self.layers:
            x = layer(x)
        x = x.tanh().squeeze(0)

        for w, shared_muscles in zip(x, self.all_muscles):
            w = w.view(1)
            mu_pair = [0.5 * (w.abs() - w), 0.5 * (w.abs() + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')

                for mu, muscle in zip(mu_pair, muscle_pair):
                    prev_a_cord = self.prev_a_clean[pointer:pointer + len(muscle)]
                    a_cord = torch.cat([mu, prev_a_cord[:-1]])
                    a_clean.append(a_cord)

                    weight = torch.linspace(0.5, 1.0, len(muscle), requires_grad=False)
                    a_cord = weight * a_cord
                    a.append(a_cord)
                    pointer += len(muscle)
        a = torch.cat(a)
        self.prev_a_clean = torch.cat(a_clean)
        return 1 - a


class OpenController(Controller):
    def __init__(self, deformable, ctrl_num):
        super(OpenController, self).__init__(deformable)
        self.ctrl_num = ctrl_num
        self.weight = nn.Parameter(torch.Tensor(*self.weight_shape()))

    def update_weight(self, weight):
        self.weight.data.copy_(weight.view(*self.weight_shape()))

    def get_grad(self):
        return self.weight.grad

    def weight_shape(self) -> tuple:
        raise NotImplementedError

    def forward(self, step, prev_a):
        raise NotImplementedError


class SharedOpenController(OpenController):
    def weight_shape(self) -> tuple:
        return (self.ctrl_num, len(self.all_muscles))

    def forward(self, step, prev_a):
        a = []
        for w, shared_muscles in zip(self.weight[step], self.all_muscles):
            for muscle_pair in shared_muscles:
                if len(muscle_pair) == 1:
                    a.append(w.sigmoid().expand(len(muscle)))
                elif len(muscle_pair) == 2:

                    # map (-1, 1) to either (-1, 0) or (0, 1)
                    w = w.tanh()
                    a.append(0.5 * (w.abs() - w))
                    a.append(0.5 * (w.abs() + w))
                else:
                    raise ValueError(f'invalid # muscle pair: {len(muscle_pair)}')
        a = torch.cat(a)
        return 1 - a


class AdaOpenController(OpenController):
    def __init__(self, deformable, ctrl_num):
        super(AdaOpenController, self).__init__(deformable, ctrl_num)
        self.a_init = torch.zeros(self.act_dofs, requires_grad=False)

    def weight_shape(self) -> tuple:
        return (self.ctrl_num, len(self.all_muscles))

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0.0, 1.0)

    def forward(self, step, prev_a) -> torch.Tensor:
        if prev_a is None:
            prev_a = self.a_init
        else:
            prev_a = 1 - prev_a
        a = []
        pointer = 0
        weight = self.weight[step].tanh()
        for w, shared_muscles in zip(weight, self.all_muscles):
            w = w.view(1)
            mu_pair = [0.5 * (w.abs() - w), 0.5 * (w.abs() + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')
                for mu, muscle in zip(mu_pair, muscle_pair):
                    prev_a_cord = prev_a[pointer:pointer + len(muscle)]
                    pointer += len(muscle)
                    a_cord = torch.cat([mu, prev_a_cord[:-1]])
                    a.append(a_cord)
        a = torch.cat(a)
        return 1 - a


class SnakeAdaOpenController(AdaOpenController):
    def __init__(self, deformable, ctrl_num):
        super(SnakeAdaOpenController, self).__init__(deformable, ctrl_num)
        self.prev_a_clean = None

    def forward(self, step, prev_a) -> torch.Tensor:
        if prev_a is None:
            self.prev_a_clean = self.a_init
        a = []
        a_clean = []
        pointer = 0
        weight = self.weight[step].tanh()
        for w, shared_muscles in zip(weight, self.all_muscles):
            w = w.view(1)
            mu_pair = [0.5 * (w.abs() - w), 0.5 * (w.abs() + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')
                for mu, muscle in zip(mu_pair, muscle_pair):
                    prev_a_cord = self.prev_a_clean[pointer:pointer + len(muscle)]
                    a_cord = torch.cat([mu, prev_a_cord[:-1]])
                    a_clean.append(a_cord)

                    weight = torch.linspace(0.5, 1.0, len(muscle), requires_grad=False)
                    a_cord = weight * a_cord
                    a.append(a_cord)
                    pointer += len(muscle)
        a = torch.cat(a)
        self.prev_a_clean = torch.cat(a_clean)
        return 1 - a
