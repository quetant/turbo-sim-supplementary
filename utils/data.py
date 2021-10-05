import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset


def normalize(data):
  mean = data.mean()
  std = data.std()
  data = (data-mean)/std
  return data, mean, std


def get_data(filename, MET=False):
  with h5py.File(filename) as f:
    data_z = f.get('FDL')[()]
    data_x = f.get('ROL')[()]

    columns_z = ['px_e-', 'py_e-', 'pz_e-', 'E_e-',
                 'px_nubar', 'py_nubar', 'pz_nubar', 'E_nubar',
                 'px_b', 'py_b', 'pz_b', 'E_b',
                 'px_bbar', 'py_bbar', 'pz_bbar', 'E_bbar',
                 'px_u', 'py_u', 'pz_u', 'E_u',
                 'px_dbar', 'py_dbar', 'pz_dbar', 'E_dbar']

    columns_x = ['px_e-', 'py_e-', 'pz_e-', 'E_e-',
                 'px_met', 'py_met', 'pz_met', 'E_met',
                 'px_jet1', 'py_jet1', 'pz_jet1', 'E_jet1',
                 'px_jet2', 'py_jet2', 'pz_jet2', 'E_jet2',
                 'px_jet3', 'py_jet3', 'pz_jet3', 'E_jet3',
                 'px_jet4', 'py_jet4', 'pz_jet4', 'E_jet4']

    data_z = pd.DataFrame(data_z, columns=columns_z)
    data_x = pd.DataFrame(data_x, columns=columns_x)

    return data_x, data_z


class PairedDataset(Dataset):
  def __init__(self, x, z, shuffle=False, decouple=False):
    super(PairedDataset, self).__init__()

    self.length = min([len(x), len(z)])
    self.x = torch.tensor(x[:self.length])
    self.z = torch.tensor(z[:self.length])

    if decouple:
      perm = torch.randperm(self.x.size(0))
      self.x = self.x[perm]

    if shuffle:
      perm = torch.randperm(self.x.size(0))
      self.x = self.x[perm]
      self.z = self.z[perm]

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    return self.x[index], self.z[index]


class UnpairedDataset(Dataset):
  def __init__(self, x, shuffle=False):
    super(UnpairedDataset, self).__init__()

    self.length = len(x)
    self.x = torch.tensor(x)

    if shuffle:
      perm = torch.randperm(self.x.size(0))
      self.x = self.x[perm]

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    return self.x[index]