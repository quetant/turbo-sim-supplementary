import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils.reconstruction as R
import utils.model as M


## Generic histogram plot
def _plot_hist(data_list,
                nbins=20, xrange=None, min_count=-1, use_log=False,
                labels=None, xlabel=None,
                show=False, save=False, path='', figname=''):
  '''
  The first data of the list is plotted with filled bars.
  The remaining ones are plotted with steps.
  '''

  bin_min, bin_max = get_min_max(data_list[0],
                                 nbins, min_count, xrange)

  gs_kw = dict(
    # width_ratios=np.ones(dims[1]),
    height_ratios=(2, 1),
    hspace=0.03,
  )
  fig, axs = plt.subplots(2, 1, figsize=(5, 7.5), squeeze=False,
                          gridspec_kw=gs_kw)

  counts_ref, bins_ref, _ = axs[0, 0].hist(
    data_list[0],
    bins=nbins, range=(bin_min, bin_max), log=use_log,
    label=labels[0], alpha=0.5
  )

  x_values = [0.5*(bins_ref[i+1]+bins_ref[i]) for i in range(len(counts_ref))]

  axs[1, 0].plot(x_values, np.ones_like(counts_ref), '--')

  for i in range(1, len(data_list)):
    counts, _, _ = axs[0, 0].hist(
      data_list[i],
      bins=nbins, range=(bin_min, bin_max), log=use_log,
      label=labels[i], histtype='step'
    )

    y_values = (counts + 1e-9)/(counts_ref + 1e-9)
    y_errors = np.sqrt(counts)/(counts + 1e-9) \
             + np.sqrt(counts_ref)/(counts_ref + 1e-9)
    y_errors *= y_values

    axs[1, 0].errorbar(x=x_values, y=y_values, yerr=y_errors, fmt='o')

  axs[0, 0].set_ylabel('count', fontsize='x-large')
  axs[0, 0].legend(fontsize='large')
  axs[0, 0].tick_params(axis='x', bottom=False)
  axs[0, 0].tick_params(axis='y', direction='in', labelsize='medium')
  axs[0, 0].ticklabel_format(axis='y', scilimits=(4, 4), useMathText=True)

  axs[1, 0].set_xlabel(xlabel, fontsize='x-large')
  axs[1, 0].set_ylabel('ratio', fontsize='x-large')
  axs[1, 0].set_ylim(0.4, 1.6)
  axs[1, 0].tick_params(direction='in', labelsize='medium')

  fig.patch.set_alpha(1.)
  if save: plt.savefig(path + figname + '.png',
                       bbox_inches='tight', dpi=300)
  if show: plt.show()
  plt.close()


def get_min_max(data, nbins, min_count, xrange):
  if min_count < 0:
    min_count = len(data) // 200

  if xrange == None:
    counts, bins = np.histogram(data, bins=nbins)
    keep = counts > min_count
    bins = bins[:-1][keep]
    bin_min, bin_max = bins.min(), bins.max()
  else:
    bin_min, bin_max = xrange

  return bin_min, bin_max


## Generic 2D histogram function
def _plot_2Dhist(X, Y,
                 nbins=20, lim=(0, 1),
                 title=None, xlabel=None, ylabel=None,
                 show=False, save=False, path='', figname=''):

  sns.set_theme(style='ticks', font_scale=1.2)

  edges = np.linspace(*lim, nbins+1)
  centers = 0.5 * (edges[1:] + edges[:-1])

  counts, _, _ = np.histogram2d(Y, X, edges)  ## /!\ Y vs X here

  import matplotlib.colors
  cmap = matplotlib.colors.ListedColormap(
    sns.color_palette('husl', nbins).as_hex()
  )

  g = sns.JointGrid(ratio=2, xlim=lim, ylim=lim, height=8, space=0.1)

  g.ax_joint.imshow(
    counts,
    extent=lim+lim, origin='lower', interpolation='none',
    cmap='gray_r'
  )

  idx = np.arange(nbins)
  for i in idx:
    g.ax_marg_x.hist(
      centers, edges, weights=np.sum(counts, axis=0)*(idx == i),
      histtype='bar', color=cmap(i)
    )

  centers_list = [centers for _ in idx]
  weights_list = [counts[:, i] for i in idx]
  color_list = [cmap(i) for i in idx]
  g.ax_marg_y.hist(
    centers_list, edges, weights=weights_list, color=color_list,
    histtype='barstacked', orientation='horizontal'
  )

  g.ax_joint.grid(False)
  g.ax_marg_x.grid(False)
  g.ax_marg_y.grid(False)

  g.fig.suptitle(title)
  g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)

  g.fig.patch.set_alpha(1.)
  if save: plt.savefig(path + figname + '.png',
                       bbox_inches='tight', dpi=300)
  if show: plt.show()
  plt.close()


## Plot results
def plot_hists(model, xi, zi,
               mean_x=0., std_x=1.,
               mean_z=0., std_z=1.,
               show=False, save=False, path=''):

  xi, zi, xt, zt, xh, zh = M.get_model_outputs(model, xi, zi,
                                               mean_x, std_x,
                                               mean_z, std_z)

  labels_x, labels_z = _get_labels()
  xlabels_x, xlabels_z = _get_xlabels()
  xranges = _get_ranges()

  # use_log = False
  nbins = 20
  # min_count = 1000
  for i in range(zi.shape[1]):
    data_list = [zi[:, i], zt[:, i], zh[:, i]]
    _plot_hist(data_list,
                nbins=nbins, xrange=xranges[i],
                labels=labels_z, xlabel=xlabels_z[i],
                show=show, save=save, path=path,
                figname=f'z_{i}')

  for i in range(xi.shape[1]):
  # for i in [11]:
    data_list = [xi[:, i], xt[:, i], xh[:, i]]
    _plot_hist(data_list,
                nbins=nbins, xrange=xranges[i],
                labels=labels_x, xlabel=xlabels_x[i],
                show=show, save=save, path=path,
                figname=f'x_{i}')


## Plot physical derived quantities
def plot_hists_reco(model, xi, zi,
                    mean_x=0., std_x=1.,
                    mean_z=0., std_z=1.,
                    show=False, save=False, path=''):

  xi, zi, xt, zt, xh, zh = M.get_model_outputs(model, xi, zi,
                                               mean_x, std_x,
                                               mean_z, std_z)

  reco_xi = R.reco(xi)
  reco_zi = R.reco(zi)
  reco_xt = R.reco(xt)
  reco_zt = R.reco(zt)
  reco_xh = R.reco(xh)
  reco_zh = R.reco(zh)

  labels_x, labels_z = _get_labels()
  xlabels = _get_xlabels_reco()
  xranges = _get_ranges_reco()
  
  # use_log = False
  nbins = 20
  # min_count = len(zi) // 100
  for k in reco_zi.keys():
    data_list = [reco_zi[k], reco_zt[k], reco_zh[k]]
    _plot_hist(data_list,
                nbins=nbins, xrange=xranges[k],
                labels=labels_z, xlabel=xlabels[k],
                show=show, save=save, path=path,
                figname=f'z_reco_{k}')
  
  for k in reco_xi.keys():
    data_list = [reco_xi[k], reco_xt[k], reco_xh[k]]
    _plot_hist(data_list,
                nbins=nbins, xrange=xranges[k],
                labels=labels_x, xlabel=xlabels[k],
                show=show, save=save, path=path,
                figname=f'x_reco_{k}')


## Plot 2D correlations between Z and X space
def plot_2D(model, xi, zi,
            mean_x=0., std_x=1.,
            mean_z=0., std_z=1.,
            show=False, save=False, path=''):

  xi, zi, xt, zt, xh, zh = M.get_model_outputs(model, xi, zi,
                                               mean_x, std_x,
                                               mean_z, std_z)

  _plot_2Dhist(
    X=zi[:, 11], Y=xi[:, 11],
    nbins=20, lim=(0, 800),
    title='Ground truth',
    xlabel='b quark energy in Z space [GeV]',
    ylabel='leading jet energy in X space [GeV]',
    show=show, save=save, path=path,
    figname='2D_Eb_truth'
  )

  _plot_2Dhist(
    X=zi[:, 11], Y=xt[:, 11],
    nbins=20, lim=(0, 800),
    title=r'Simulation $z \rightarrow \tilde{x}$',
    xlabel='b quark energy in Z space [GeV]',
    ylabel='leading jet energy in X space [GeV]',
    show=show, save=save, path=path,
    figname='2D_Eb_sim'
  )


## Give LaTeX-style labels
def _get_labels():
  labels_z = [
    '$z$',
    r'$x \rightarrow \tilde{z}$',
    r'$z \rightarrow \tilde{x} \rightarrow \hat{z}$'
  ]

  labels_x = [
    '$x$',
    r'$z \rightarrow \tilde{x}$',
    r'$x \rightarrow \tilde{z} \rightarrow \hat{x}$'
  ]

  return labels_x, labels_z


def _get_xlabels():
  xlabels_z = [
    '$p_x^{e^-}$', '$p_y^{e^-}$', '$p_z^{e^-}$', '$E^{e^-}$',
    r'$p_x^{\bar{\nu}}$', r'$p_y^{\bar{\nu}}$', r'$p_z^{\bar{\nu}}$', r'$E^{\bar{\nu}}$',
    '$p_x^b$', '$p_y^b$', '$p_z^b$', '$E^b$',
    r'$p_x^{\bar{b}}$', r'$p_y^{\bar{b}}$', r'$p_z^{\bar{b}}$', r'$E^{\bar{b}}$',
    '$p_x^u$', '$p_y^u$', '$p_z^u$', '$E^u$',
    r'$p_x^{\bar{d}}$', r'$p_y^{\bar{d}}$', r'$p_z^{\bar{d}}$', r'$E^{\bar{d}}$'
  ]

  xlabels_x = [
    '$p_x^{e^-}$', '$p_y^{e^-}$', '$p_z^{e^-}$', '$E^{e^-}$',
    '$p_x^{miss}$', '$p_y^{miss}$', '$p_z^{miss}$', '$E^{miss}$',
    '$p_x^{jet1}$', '$p_y^{jet1}$', '$p_z^{jet1}$', '$E^{jet1}$',
    '$p_x^{jet2}$', '$p_y^{jet2}$', '$p_z^{jet2}$', '$E^{jet2}$',
    '$p_x^{jet3}$', '$p_y^{jet3}$', '$p_z^{jet3}$', '$E^{jet3}$',
    '$p_x^{jet4}$', '$p_y^{jet4}$', '$p_z^{jet4}$', '$E^{jet4}$'
  ]

  xlabels_z = [s + ' [GeV]' for s in xlabels_z]
  xlabels_x = [s + ' [GeV]' for s in xlabels_x]

  return xlabels_x, xlabels_z


def _get_xlabels_reco():
  xlabels = {
    'mw_lep': '$m_W^{lep}$ [GeV]',
    'mt_lep': '$m_t^{lep}$ [GeV]',
    'mw_had': '$m_W^{had}$ [GeV]',
    'mt_had': '$m_t^{had}$ [GeV]',
    'mtt': '$m_{tt}$ [GeV]',
  }
  return xlabels


## Specify the range of the plots
def _get_ranges():
  ranges = [
    (-150, 150),
    (-150, 150),
    (-500, 500),
    (0, 500),
    (-150, 150),
    (-150, 150),
    (-3000, 3000),
    (0, 3000),
    (-250, 250),
    (-250, 250),
    (-1250, 1250),
    (0, 1250),
    (-250, 250),
    (-250, 250),
    (-1250, 1250),
    (0, 1250),
    (-250, 250),
    (-250, 250),
    (-1250, 1250),
    (0, 1250),
    (-250, 250),
    (-250, 250),
    (-1250, 1250),
    (0, 1250),
  ]
  return ranges


def _get_ranges_reco():
  ranges = {
    'mtt': (250, 2500),
    'mw_lep': (20, 200),
    'mw_had': (20, 200),
    'mt_lep': (80, 400),
    'mt_had': (50, 900),
  }
  return ranges


## Losses plot
def plot_losses(loss_evol, keys=[], n_epochs=0, n_batch=0,
                show=False, save=False, path=''):
  loss_evol_mean = {}
  for k in keys:

    ## Plot training losses
    loss_evol_mean = _compute_mean_loss(
      loss_evol, loss_evol_mean,
      key=k, n_epochs=n_epochs, n_batch=n_batch
    )
    p = plt.plot(loss_evol_mean[k], marker='.', ls='', alpha=0.5, label=k)

    ## Plot validation losses
    k += '_valid'
    loss_evol_mean = _compute_mean_loss(
      loss_evol, loss_evol_mean,
      key=k, n_epochs=n_epochs, n_batch=n_batch
    )
    plt.plot(loss_evol_mean[k], ls='--', alpha=0.5, label=k,
              c=p[0].get_color())

  plt.legend()
  # plt.ylim(0, 0.1e19)
  if save: plt.savefig(path + '_losses.png', bbox_inches='tight', dpi=300)
  if show: plt.show()
  plt.close()


def _compute_mean_loss(loss_evol, loss_evol_mean, key, n_epochs, n_batch):
  loss_evol_mean[key] = []
  for i in range(n_epochs):
    loss_evol_mean[key].append(
        np.mean([l for l in loss_evol[key][n_batch*i:n_batch*(i+1)]])
    )
  return loss_evol_mean