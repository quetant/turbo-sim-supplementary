import numpy as np
from numpy.polynomial import Polynomial


def reco(data):
  mw_leps = []
  mt_leps = []
  mw_hads = []
  mt_hads = []
  mtts = []
  for event in data:
    mw_lep, mt_lep, mw_had, mt_had, mtt= _reco_event(event)
    mw_leps.append(mw_lep)
    mt_leps.append(mt_lep)
    mw_hads.append(mw_had)
    mt_hads.append(mt_had)
    mtts.append(mtt)

  reco = {
    'mw_lep': np.array(mw_leps).flatten(),
    'mt_lep': np.array(mt_leps).flatten(),
    'mw_had': np.array(mw_hads).flatten(),
    'mt_had': np.array(mt_hads).flatten(),
    'mtt': np.array(mtts).flatten(),
  }
  return reco


def _mass_from_p4(p4):
  p4 = p4.reshape(-1, 4)
  m2 = p4[:, -1]**2 - np.sum(p4[:, :-1]**2, axis=1)
  mask = m2 < 0.
  sign = (-1)**mask
  # m2 = np.where(m2 < 0., -m2, m2)
  m = sign * np.sqrt(sign*m2)
  return m


def _reco_event(event):
  mw_true = 80.379  # GeV (https://pdg.lbl.gov/2020/tables/rpp2020-sum-gauge-higgs-bosons.pdf)
  mt_true = 172.76  # GeV (https://pdg.lbl.gov/2020/tables/rpp2020-sum-quarks.pdf)

  p4_e    = event[0:4]
  p4_met  = event[4:8]
  p4_nu = _reco_neutrino(p4_e, p4_met, mw_true)
  p4_jet1 = event[8:12]
  p4_jet2 = event[12:16]
  p4_jet3 = event[16:20]
  p4_jet4 = event[20:24]
  p4_jets = [p4_jet1, p4_jet2, p4_jet3, p4_jet4]

  test_best = -1
  i1_list = [0, 1, 2, 3]
  for i1 in i1_list:
    mt_lep = _mass_from_p4(p4_e + p4_nu + p4_jets[i1])
    i2_list = i1_list.copy()
    i2_list.remove(i1)
    for i2 in i2_list:
      i3_list = i2_list.copy()
      i3_list.remove(i2)
      for i3 in i3_list:
        if i3 < i2:
          mw_had = _mass_from_p4(p4_jets[i2] + p4_jets[i3])
          i4_list = i3_list.copy()
          i4_list.remove(i3)
          for i4 in i4_list:
            mt_had = _mass_from_p4(p4_jets[i2] + p4_jets[i3] + p4_jets[i4])

            test = ((mt_lep-mt_true)/5.)**2 + ((mw_had-mw_true)/10.)**2 \
                 + ((mt_had-mt_true)/15.)**2

            if test < test_best or test_best < 0:
              test_best = test
              idx = (i1, i2, i3, i4)
  
  i1, i2, i3, i4 = idx
  mw_lep = _mass_from_p4(p4_e + p4_nu)
  mt_lep = _mass_from_p4(p4_e + p4_nu + p4_jets[i1])
  mw_had = _mass_from_p4(p4_jets[i2] + p4_jets[i3])
  mt_had = _mass_from_p4(p4_jets[i2] + p4_jets[i3] + p4_jets[i4])
  mtt = _mass_from_p4(p4_e + p4_nu + p4_jet1 + p4_jet2 + p4_jet3 + p4_jet4)

  return mw_lep, mt_lep, mw_had, mt_had, mtt


def _reco_neutrino(p4_e, p4_met, mw_true):
  '''
  From https://github.com/yiboyang/otus/blob/main/utilityFunctions/top_masses.py

  Solution of mw^2 = (p_e + p_nu)^2 (in 4-vector)
    with p_nu = (E_nu, px_met, py_met, pz_nu)
    -> solve it for pz_nu
  '''

  me = _mass_from_p4(p4_e)
  px_e = p4_e[0]
  py_e = p4_e[1]
  pz_e = p4_e[2]
  E_e = p4_e[3]
  px_met = p4_met[0]
  py_met = p4_met[1]

  k = ((mw_true**2 - me**2) / 2) + (px_e*px_met + py_e*py_met)
  
  a = E_e**2 - pz_e**2
  b = -2 * k * pz_e
  c = E_e**2 * (px_met**2 + py_met**2) - k**2

  poly = Polynomial([c.item(), b.item(), a.item()])
  pz_nu = min(poly.roots(), key=abs).real

  p4_nu = np.array([
    p4_met[0],
    p4_met[1],
    pz_nu.item(),
    np.sqrt(p4_met[0]**2 + p4_met[1]**2 + pz_nu**2).item()
  ])

  return p4_nu