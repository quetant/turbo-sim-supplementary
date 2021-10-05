import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDLoss(nn.Module):
  def __init__(self, alpha):
    super(MMDLoss, self).__init__()

    self.alpha = alpha

  def forward(self, x, y):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    # Why are they called K, L and P?
    K = torch.exp(-self.alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(-self.alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(-self.alpha * (rx.t() + ry - 2*xy))

    return K.sum() + L.sum() - 2*P.sum()


class DRELoss(nn.Module):
  def __init__(self):
    super(DRELoss, self).__init__()

  def forward(self, preds_real, preds_fake):
    targets_real = torch.ones_like(preds_real)
    targets_fake = torch.zeros_like(preds_fake)
    loss_real = F.binary_cross_entropy(preds_real, targets_real)
    loss_fake = F.binary_cross_entropy(preds_fake, targets_fake)
    return loss_real + loss_fake


def get_wgp_interp(real, fake):
  alpha = torch.rand(
    size=(real.size(0), 1),
    device=real.device
  ).expand((real.size(0), real.size(1)))
  interp = alpha*real + (1-alpha)*fake
  interp = interp.detach()
  interp.requires_grad = True
  return interp


class WGPLoss(nn.Module):
  def __init__(self):
    super(WGPLoss, self).__init__()

  def forward(self, preds_real, preds_fake, preds_interp, interp, weight_grad):
    loss = preds_fake - preds_real
    penalty = weight_grad*self._calc_grad(preds_interp, interp)
    loss += penalty
    return loss.mean()

  def _calc_grad(self, preds_interp, interp):
    gradient = torch.autograd.grad(
      outputs=preds_interp, inputs=interp,
      grad_outputs=torch.ones_like(preds_interp),
      create_graph=True,
      retain_graph=True,
      only_inputs=True
    )[0]
    penalty = gradient.norm(2, dim=1).view(-1, 1)
    penalty = (penalty - 1)**2
    return penalty


class CriticLoss(nn.Module):
  def __init__(self, mode):
    super(CriticLoss, self).__init__()
    self.mode = mode

  def forward(self, preds_fake):
    if self.mode == 'DRE':
      targets_real = torch.ones_like(preds_fake)
      loss = F.binary_cross_entropy(preds_fake, targets_real)
      return loss.mean()
    elif self.mode == 'WGP':
      loss = -preds_fake # Have to check what to return
      return loss.mean()