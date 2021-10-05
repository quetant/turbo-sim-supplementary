import torch
import torch.nn as nn
import utils.losses as L


def count_parameters(model):
  return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_activation_function(activation='relu'):
  if activation == 'relu':
    return nn.ReLU()
  elif activation == 'leaky_relu':
    return nn.LeakyReLU()
  elif activation == 'sigmoid':
    return nn.Sigmoid()
  elif activation == 'tanh':
    return nn.Tanh()


def get_loss_function(mode='DRE'):
  if mode == 'DRE':
    return L.DRELoss()
  elif mode == 'WGP':
    return L.WGPLoss()


## Load a model
def load_model(model, critics, which='', path='.'):
  model.load_state_dict(torch.load(path + f'_{which}.pt'))
  model.eval()
  for k in critics.keys():
    critics[k].load_state_dict(torch.load(path + f'_{k}_{which}.pt'))
    critics[k].eval()
  return model, critics


## Get the model outputs
def get_model_outputs(model, xi, zi,
                      mean_x, std_x,
                      mean_z, std_z):

  with torch.no_grad():
    device = model.device
    model.eval()

    xi = torch.tensor(xi).to(device)
    zi = torch.tensor(zi).to(device)

    xh, zt = model(x=xi)
    xt, zh = model(z=zi, order='reverse')

    xi = xi.detach().cpu().numpy() * std_x + mean_x
    zi = zi.detach().cpu().numpy() * std_z + mean_z

    xt = xt.detach().cpu().numpy() * std_x + mean_x
    zt = zt.detach().cpu().numpy() * std_z + mean_z

    xh = xh.detach().cpu().numpy() * std_x + mean_x
    zh = zh.detach().cpu().numpy() * std_z + mean_z

  return xi, zi, xt, zt, xh, zh