import torch
import torch.nn as nn
import utils.model as M
import utils.losses as L


class MLP(nn.Module):
  def __init__(self, dim_in, dim_out, dim_hid, act='relu', batch_norm=False):
    super(MLP, self).__init__()

    self.dim_in = dim_in
    self.dim_out = dim_out
    self.dim_hid = dim_hid
    self.act_fn = M.get_activation_function(act)
    self.batch_norm = batch_norm

    dims = [dim_in] + dim_hid + [dim_out]
    self.layers = nn.ModuleList(
      [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
    )

    if batch_norm:
      self.layers_bn = nn.ModuleList(
        [nn.BatchNorm1d(dims[i+1]) for i in range(len(dims)-2)]
      )

  def forward(self, x):
    for i in range(len(self.layers[:-1])):
      x = self.layers[i](x)
      if self.batch_norm: x = self.layers_bn[i](x)
      x = self.act_fn(x)
    x = self.layers[-1](x)
    return x


class Critic(nn.Module):
  def __init__(self, dim_in, dim_out, dim_hid=[],
               act='relu', mode='DRE', weight_grad=1.,
               device='cpu'):
    super(Critic, self).__init__()

    self.dim_in = dim_in
    self.dim_out = dim_out
    self.mode = mode
    self.weight_grad = weight_grad
    self.device = device

    self.train_loss_fn = M.get_loss_function(mode)
    self.loss_fn = L.CriticLoss(mode)

    self.network = MLP(dim_in, dim_out, dim_hid, act)

  def forward(self, x):
    out = self.network(x)
    if self.mode == 'DRE':
      out = torch.sigmoid(out)
    return out

  def training_loss(self, real, fake):
    if self.mode == 'DRE':
      preds_real = self(real)
      preds_fake = self(fake)
      loss = self.train_loss_fn(preds_real, preds_fake)
    elif self.mode == 'WGP':
      preds_real = self(real)
      preds_fake = self(fake)
      interp = L.get_wgp_interp(real, fake)
      preds_interp = self(interp)
      loss = self.train_loss_fn(preds_real, preds_fake,
                                preds_interp, interp, self.weight_grad)
    return loss

  def loss(self, fake):
    preds_fake = self(fake)
    loss = self.loss_fn(preds_fake)
    return loss