import torch
import torch.nn as nn
import models.base


class TurboSim(nn.Module):
  def __init__(self, dim_x, dim_z, dim_en=[], dim_de=[],
               act_en='relu', act_de='relu',
               batch_norm=False,
               stochastic=True, stochastic_mode='add',
               device='cpu'):

    super(TurboSim, self).__init__()

    if stochastic and stochastic_mode == 'cat':
      dim_x *= 2
      dim_z *= 2

    self.dim_x = dim_x
    self.dim_z = dim_z
    self.dim_en = dim_en
    self.dim_de = dim_de
    self.stochastic = stochastic
    self.stochastic_mode = stochastic_mode
    self.device = device

    self.encoder = models.base.MLP(dim_x, dim_z, dim_en, act_en, batch_norm)
    self.decoder = models.base.MLP(dim_z, dim_x, dim_de, act_de, batch_norm)

  def forward(self, x=None, z=None, order='direct'):
    if order == 'direct':
      z = self.encode(x)
      x = self.decode(z)
    elif order == 'reverse':
      x = self.decode(z)
      z = self.encode(x)
    return x, z

  def encode(self, x):
    if self.stochastic: x = self._apply_noise(x)
    z = self.encoder(x)
    return z

  def decode(self, z):
    if self.stochastic: z = self._apply_noise(z)
    x = self.decoder(z)
    return x

  def _apply_noise(self, input):
    if self.stochastic_mode == 'cat':
      return self._cat_noise(input)
    elif self.stochastic_mode == 'mul':
      return self._mul_noise(input)
    elif self.stochastic_mode == 'add':
      return self._add_noise(input)

  def _cat_noise(self, input):
    eps = torch.randn_like(input)
    input = torch.cat([input, eps], dim=1)
    return input
  
  def _mul_noise(self, input):
    eps = torch.randn_like(input)
    input *= eps
    return input

  def _add_noise(self, input):
    eps = torch.randn_like(input)
    input += 0.05*eps
    return input