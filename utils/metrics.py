import numpy as np
import scipy.stats
import utils.plots as P
import utils.model as M


## Chi square
def compute_chi2(model, xi, zi,
                 mean_x=0., std_x=1.,
                 mean_z=0., std_z=1.):

  xi, zi, xt, zt, xh, zh = M.get_model_outputs(model, xi, zi,
                                               mean_x, std_x,
                                               mean_z, std_z)
  nbins = 20
  chi2s = {'z': [], 'x': []}
  for i in range(zi.shape[1]):
    data_list = [zi[:, i], zt[:, i], zh[:, i]]
    chi2s['z'].append(_compute_chi2(data_list, nbins))

  for i in range(xi.shape[1]):
    data_list = [xi[:, i], xt[:, i], xh[:, i]]
    chi2s['x'].append(_compute_chi2(data_list, nbins))

  return chi2s


def _compute_chi2(data_list,
                  nbins):
  bin_min, bin_max = P.get_min_max(data_list[0],
                                   nbins=nbins, min_count=-1, xrange=None)
 
  counts_ref, _ = np.histogram(
      data_list[0], bins=nbins, range=(bin_min, bin_max)
    )

  chi2 = {}
  keys = ['tilde', 'hat']
  for j in range(1, len(data_list)):
    counts, _ = np.histogram(
      data_list[j], bins=nbins, range=(bin_min, bin_max)
    )

    c = 0.
    for b in range(len(counts)):
      c += (counts[b] - counts_ref[b])**2 / counts_ref[b]  ## Chi2 computation

    chi2[keys[j-1]] = c / (len(counts)-1)  ## Reduced Chi2

  return chi2


def compute_chi2_sum(chi2):
  chi2_sum = 0.
  for figs in chi2.values():
      for fig in figs:
          for v in fig.values():
              chi2_sum += v
  return chi2_sum


## Kolmogorov–Smirnov
def compute_ks_total(model, xi, zi,
                     mean_x=0., std_x=1.,
                     mean_z=0., std_z=1.):

  xi, zi, xt, zt, xh, zh = M.get_model_outputs(model, xi, zi,
                                               mean_x, std_x,
                                               mean_z, std_z)

  stat_z = 0.
  for i in range(zi.shape[1]):
    stat_t, _ = scipy.stats.ks_2samp(zi[:, i], zt[:, i])
    stat_h, _ = scipy.stats.ks_2samp(zi[:, i], zh[:, i])
    stat_z += stat_t + stat_h

  stat_x = 0.
  for i in range(xi.shape[1]):
    stat_t, _ = scipy.stats.ks_2samp(xi[:, i], xt[:, i])
    stat_h, _ = scipy.stats.ks_2samp(xi[:, i], xh[:, i])
    stat_x += stat_t + stat_h

  return stat_z + stat_x