import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils.data as D
import utils.metrics as metrics
import utils.misc as misc


## Turbo-Sim training
def train_turbo(model, opt_model,
                critics, opt_critics,
                data_x_train, data_z_train,
                data_x_valid, data_z_valid,
                mean_x, std_x, mean_z, std_z,
                weights,
                epochs=1, batch_size=1, start_from=0,
                grad_clip=False,
                early_stop=False,
                save=False, path='', output_model='model',
                device='cpu'):

  losses_previous = []
  n_previous = epochs // 10
  loss_best = -1.
  loss_best_reco = -1.
  ks_best = -1.
  epoch_best = 0
  epoch_best_reco = 0
  epoch_best_ks = 0
  n_valid = len(data_x_valid)

  model = model.to(device)
  model.device = device
  model.train()
  for k in critics.keys():
    critics[k] = critics[k].to(device)
    critics[k].train()

  loss_evol = {}
  for key, weight in weights.items():
    if weight != 0.:
      loss_evol[key] = []
      loss_evol[key + '_valid'] = []

  for epoch in range(start_from, epochs):
    time_start = time.time()
    train_ds = D.PairedDataset(data_x_train, data_z_train, decouple=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    loss_valid_total = 0.
    loss_valid_total_reco = 0.
    loss_previous = 0.
    i = -1
    for _, (xi, zi) in enumerate(train_dl):
      xi = xi.to(device)
      zi = zi.to(device)
      xh, zt = model(x=xi)
      xt, zh = model(z=zi, order='reverse')

      ## Validation data (not used for training)
      if batch_size*(i+1) >= n_valid or i < 0:
        i = 0
        valid_ds = D.PairedDataset(data_x_valid, data_z_valid,
                                   decouple=True, shuffle=True)
      else:
        i += 1
      xi_valid, zi_valid = valid_ds[batch_size*i:batch_size*(i+1)]
      xi_valid = xi_valid.to(device)
      zi_valid = zi_valid.to(device)
      xh_valid, zt_valid = model(x=xi_valid)
      xt_valid, zh_valid = model(z=zi_valid, order='reverse')
      ####

      critics, opt_critics = _train_critics(critics, opt_critics, weights,
                                            xi=xi, xt=xt, xh=xh,
                                            zi=zi, zt=zt, zh=zh)

      opt_model.zero_grad()
      loss, loss_evol = _compute_loss(critics, weights,
                                      xi=xi, xt=xt, xh=xh,
                                      zi=zi, zt=zt, zh=zh,
                                      loss_evol=loss_evol,
                                      device=device)

      loss_previous += loss.item()

      ## Validation loss (not used for training)
      loss_valid, loss_valid_critics, loss_evol = _compute_loss(
        critics, weights,
        xi=xi_valid, xt=xt_valid, xh=xh_valid,
        zi=zi_valid, zt=zt_valid, zh=zh_valid,
        loss_evol=loss_evol,
        valid=True, device=device
      )
      loss_valid_total += loss_valid.item() + loss_valid_critics.item()
      loss_valid_total_reco += loss_valid.item()
      ####

      loss.backward()
      if grad_clip > 0.:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
      opt_model.step()

    ## Validation metrics
    ks_valid_total = metrics.compute_ks_total(
      model,
      xi=data_x_valid, zi=data_z_valid,
      mean_x=mean_x, std_x=std_x,
      mean_z=mean_z, std_z=std_z
    )
    ####

    if save:
      if loss_valid_total < loss_best or loss_best < 0.:
        loss_best = loss_valid_total
        epoch_best = epoch
        torch.save(model.state_dict(), path + output_model + '_best.pt')
        for k in critics.keys():
          torch.save(critics[k].state_dict(),
                      path + output_model + f'_{k}_best.pt')

      if loss_valid_total_reco < loss_best_reco or loss_best_reco < 0.:
        loss_best_reco = loss_valid_total_reco
        epoch_best_reco = epoch
        torch.save(model.state_dict(), path + output_model + '_best_reco.pt')
        for k in critics.keys():
          torch.save(critics[k].state_dict(),
                      path + output_model + f'_{k}_best_reco.pt')

      if ks_valid_total < ks_best or ks_best < 0.:
        ks_best = ks_valid_total
        epoch_best_ks = epoch
        torch.save(model.state_dict(), path + output_model + '_best_ks.pt')
        for k in critics.keys():
          torch.save(critics[k].state_dict(),
                      path + output_model + f'_{k}_best_ks.pt')

    time_end = time.time()
    print(f'{epoch+1}/{epochs}: {time_end-time_start:.2f}s')

    losses_previous.append(loss_previous)
    if early_stop and len(losses_previous) >= n_previous and \
      abs(np.mean(losses_previous[-n_previous:])/loss_previous - 1) < 0.001:
        print('Early stopped!')
        break

  if save:
    with open(path + 'log.txt', 'a') as f:
      f.write(f'best epoch: {epoch_best}\n')
      f.write(f'best loss: {loss_best}\n')
      f.write(f'best epoch reco: {epoch_best_reco}\n')
      f.write(f'best reco loss: {loss_best_reco}\n')
      f.write(f'best epoch KS: {epoch_best_ks}\n')
      f.write(f'best KS: {ks_best}\n')
      f.write(f'last epoch: {epoch}\n')

  return model, critics, loss_evol


## Generic functions
def _train_critics(critics, opt_critics, weights,
                   xi=None, xt=None, xh=None,
                   zi=None, zt=None, zh=None):
  data_real = {
    'Dxt': xi,
    'Dxh': xi,
    'Dzt': zi,
    'Dzh': zi,
  }
  data_fake = {
    'Dxt': xt,
    'Dxh': xh,
    'Dzt': zt,
    'Dzh': zh,
  }
  for key, weight in weights.items():
    if key[0] == 'D' and weight != 0.:
      opt_critics[key].zero_grad()
      loss = critics[key].training_loss(
        real=data_real[key],
        fake=data_fake[key]
      )
      loss.backward(retain_graph=True)
      opt_critics[key].step()

  return critics, opt_critics


def _compute_loss(critics, weights,
                  xi=None, xt=None, xh=None,
                  zi=None, zt=None, zh=None,
                  loss_evol=None,
                  valid=False, device='cpu'):

  data_real = {
    'dxt': xi,
    'dxh': xi,
    'dzt': zi,
    'dzh': zi,
  }
  data_fake = {
    'Dxt': xt,
    'Dxh': xh,
    'Dzt': zt,
    'Dzh': zh,
  }
  data_reco = {
    'dxt': xt,
    'dxh': xh,
    'dzt': zt,
    'dzh': zh,
  }

  suffixe = '_valid' if valid else ''

  loss_supervised = torch.tensor(0., device=device)
  loss_critics = torch.tensor(0., device=device)

  ## Critics
  for key, weight in weights.items():
    if key[0] == 'D' and weight != 0.:
      loss = critics[key].loss(fake=data_fake[key])
      loss *= weights[key]
      loss_evol[f'{key}{suffixe}'].append(loss.item())
      loss_critics += loss

  ## Supervised
  for key, weight in weights.items():
    if key[0] == 'd' and weight != 0.:
      loss = F.mse_loss(data_real[key], data_reco[key])
      loss *= weights[key]
      loss_evol[f'{key}{suffixe}'].append(loss.item())
      loss_supervised += loss

  if not valid:
    return loss+loss_critics, loss_evol
  else:
    return loss, loss_critics, loss_evol