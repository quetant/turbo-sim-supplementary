import numpy as np
import datetime
import os
import torch
import torch.nn.functional as F


def one_hot_like(data):
  target = torch.randint(
    low=0, high=data.size(1), size=(data.size(0),), device=data.device
  )
  one_hot = F.one_hot(target, num_classes=data.size(1))
  return one_hot.float()


def running_mean(x, N=1):
  cs = np.cumsum(np.insert(x, 0, 0))
  return (cs[N:] - cs[:-N]) / N


def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)
  return path


def now(path):
  dt = datetime.datetime.now()
  dt = dt - datetime.timedelta(minutes=dt.minute % 5,
                               seconds=dt.second,
                               microseconds=dt.microsecond)
  dir_name = path + dt.strftime('%Y-%m-%d_%Hh%M/')
  return mkdir(dir_name)


def write_log(
  epochs, batch_size,
  weights, weights_grad,
  activation,
  optimizer, lr, weight_decay,
  alpha, beta_1, beta_2, momentum,
  scheduler, grad_clip, batch_norm,
  early_stop,
  critics_mode,
  lr_crit, weight_decay_crit,
  model_name,
  path=''):

  lines = [
      f'model name: {model_name}\n',
      f'epochs: {epochs}\n',
      f'batch size: {batch_size}\n',
      f'activation: {activation}\n',
      f'critics mode: {critics_mode}\n',
      f'optimizer: {optimizer}\n',
      f'learning rate model: {lr}\n',
      f'learning rate critics: {lr_crit}\n',
      f'weight decay model: {weight_decay}\n',
      f'weight decay critics: {weight_decay_crit}\n',
      f'alpha: {alpha}\n',
      f'beta 1: {beta_1}\n',
      f'beta 2: {beta_2}\n',
      f'momentum: {momentum}\n',
      f'scheduler: {scheduler}\n',
      f'gradient clipping: {grad_clip}\n',
      f'batch normalization: {batch_norm}\n',
      f'early stopping: {early_stop}\n',
  ]

  lines.append('weights\n')
  for k, v in weights.items():
    lines.append(f' {k}: {v}\n')

  lines.append('weights gradients\n')
  for k, v in weights_grad.items():
    lines.append(f' {k}: {v}\n')

  with open(path + 'log.txt', 'w') as f:
    f.writelines(lines)