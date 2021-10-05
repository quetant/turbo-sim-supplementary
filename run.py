import argparse
import torch
import models.base
from models.turbo import TurboSim
import utils.data as D
import utils.train as T
import utils.plots as P
import utils.model as M
import utils.misc as misc


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--start-from-epoch", type=int, default=0)
    parser.add_argument("--critics-mode", type=str, default='WGP')
    parser.add_argument("--early-stop", type=str2bool, default=False)
    parser.add_argument("--model-name", type=str, default='turbosim')
    parser.add_argument("--path-data", type=str, default='data/')
    parser.add_argument("--path-models", type=str, default='experiment/')
    parser.add_argument("--path-images", type=str, default='experiment/')
    parser.add_argument("--load", type=str2bool, default=False)
    parser.add_argument("--plot", type=str2bool, default=True)
    parser.add_argument("--use-cuda", type=str2bool, default=True)
    parser.add_argument("--load-epoch", type=int, default=0)
    parser.add_argument("--load-which", type=str, default='final')
    parser.add_argument("--plot-reco-lim", type=int, default=10000)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--activation", type=str, default='relu')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-crit", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--momentum", type=float, default=-1.0)
    parser.add_argument("--beta-1", type=float, default=0.9)
    parser.add_argument("--beta-2", type=float, default=0.7)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--weight-decay-crit", type=float, default=1e-4)
    parser.add_argument("--batch-norm", type=str2bool, default=False)
    parser.add_argument("--grad-clip", type=float, default=-1.0)
    parser.add_argument("--w-Dzt", type=float, default=1.)
    parser.add_argument("--w-dzt", type=float, default=10.)
    parser.add_argument("--w-Dzh", type=float, default=0.)
    parser.add_argument("--w-dzh", type=float, default=100.)
    parser.add_argument("--w-Dxt", type=float, default=10.)
    parser.add_argument("--w-dxt", type=float, default=0.)
    parser.add_argument("--w-Dxh", type=float, default=1.)
    parser.add_argument("--w-dxh", type=float, default=10.)
    parser.add_argument("--w-Dzt-grad", type=float, default=1000.)
    parser.add_argument("--w-Dzh-grad", type=float, default=1000.)
    parser.add_argument("--w-Dxt-grad", type=float, default=10.)
    parser.add_argument("--w-Dxh-grad", type=float, default=1000.)

    opt = parser.parse_args()
    print(opt)

    device = 'cuda' if torch.cuda.is_available() and opt.use_cuda else 'cpu'
    print(f'Device: {device}')

    MODEL_NAME = opt.model_name
    TRAIN_NEW = not opt.load
    SAVE = True
    PLOT = opt.plot
    LOAD_EPOCH = opt.load_epoch
    LOAD_WHICH = opt.load_which
    RECO_LIM = opt.plot_reco_lim

    PATH_DATA = opt.path_data
    PATH_MODELS = misc.mkdir(opt.path_models)
    PATH_IMAGES = misc.mkdir(opt.path_images)

    input_file = 'ppttbar.hdf5'
    input_file_test = 'ppttbar_test.hdf5'
    output_model = f'{MODEL_NAME}'

    n_train = 200_000
    n_valid = 50_000
    n_test = 160_000
    BATCH_SIZE = opt.batch_size

    n_batch = n_train // BATCH_SIZE
    dim_data = 24
    dim_latent = dim_data
    N_EPOCHS = opt.n_epochs
    START_FROM_EPOCH = opt.start_from_epoch
    OPTIMIZER = opt.optimizer
    ACTIVATION = opt.activation
    LEARNING_RATE = opt.lr
    LEARNING_RATE_CRIT = opt.lr_crit
    ALPHA = opt.alpha
    MOMENTUM = opt.momentum
    BETA_1 = opt.beta_1
    BETA_2 = opt.beta_2
    WEIGHT_DECAY = opt.weight_decay
    WEIGHT_DECAY_CRIT = opt.weight_decay_crit
    BATCH_NORM = opt.batch_norm
    GRAD_CLIP = opt.grad_clip
    EARLY_STOP = opt.early_stop
    CRITICS_MODE = opt.critics_mode

    data_x, data_z = D.get_data(PATH_DATA + input_file)
    print('Full data')
    print(f'x: {data_x.shape}')
    print(f'z: {data_z.shape}')

    data_x = data_x[:n_train+n_valid].to_numpy(dtype='float32')
    data_z = data_z[:n_train+n_valid].to_numpy(dtype='float32')
    data_x, mean_x, std_x = D.normalize(data_x)
    data_z, mean_z, std_z = D.normalize(data_z)

    data_x_train = data_x[:n_train]
    data_z_train = data_z[:n_train]
    print('Training data')
    print(f'x: {data_x_train.shape}')
    print(f'z: {data_z_train.shape}')

    data_x_valid = data_x[n_train:n_train+n_valid]
    data_z_valid = data_z[n_train:n_train+n_valid]
    print('Validation data')
    print(f'x: {data_x_valid.shape}')
    print(f'z: {data_z_valid.shape}')

    data_x_test, data_z_test = D.get_data(PATH_DATA + input_file_test)
    data_x_test = data_x_test[:n_test].to_numpy(dtype='float32')
    data_z_test = data_z_test[:n_test].to_numpy(dtype='float32')
    data_x_test = (data_x_test - mean_x) / std_x
    data_z_test = (data_z_test - mean_z) / std_z
    print('Test data')
    print(f'x: {data_x_test.shape}')
    print(f'z: {data_z_test.shape}')

    ## All Turbo-Sim weights:
    ## 'd' for supervised loss, 'D' for unsupervised
    ## 't' = tilde, 'h' = hat, 'th' = tilde/hat
    weights = {
        'dzt': opt.w_dzt,
        'Dzt': opt.w_Dzt,
        'dxh': opt.w_dxh,
        'Dxh': opt.w_Dxh,
        'dxt': opt.w_dxt,
        'Dxt': opt.w_Dxt,
        'dzh': opt.w_dzh,
        'Dzh': opt.w_Dzh,
        'dzth': 0.,
        'Dzth': 0.,
        'dxth': 0.,
        'Dxth': 0.,
    }

    weights_grad = {
        'Dzt': opt.w_Dzt_grad,
        'Dzh': opt.w_Dzh_grad,
        'Dxt': opt.w_Dxt_grad,
        'Dxh': opt.w_Dxh_grad,
    }

    model = TurboSim(
        dim_x=dim_data, dim_z=dim_latent,
        dim_en=[256, 128, 64], dim_de=[64, 128, 256],
        act_en=ACTIVATION, act_de=ACTIVATION,
        batch_norm=BATCH_NORM,
        device=device
    )

    if OPTIMIZER == 'adam':
        opt_model = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE,
                                     betas=(BETA_1, BETA_2),
                                     weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == 'rmsprop':
        opt_model = torch.optim.RMSprop(model.parameters(),
                                        lr=LEARNING_RATE,
                                        alpha=ALPHA,
                                        momentum=MOMENTUM,
                                        weight_decay=WEIGHT_DECAY)

    critics = {}
    opt_critics = {}
    for k in ['Dzt', 'Dzh', 'Dxt', 'Dxh']:
        critics[k] = models.base.Critic(
            dim_in=dim_latent, dim_out=1, dim_hid=[256, 128, 64],
            mode=CRITICS_MODE, weight_grad=weights_grad[k]
        )
        opt_critics[k] = torch.optim.Adam(critics[k].parameters(),
                                          lr=LEARNING_RATE_CRIT,
                                          weight_decay=WEIGHT_DECAY_CRIT)

    print(f'Total params: {M.count_parameters(model)}')
    print(f'Train params: {M.count_trainable_parameters(model)}')
    print(model)
    print()

    for critic in critics.values():
        print(f'Total params: {M.count_parameters(critic)}')
        print(f'Train params: {M.count_trainable_parameters(critic)}')
        print(critic)
        print()

    if TRAIN_NEW:
        print()
        print('Write log...')
        misc.write_log(
            epochs=N_EPOCHS, batch_size=BATCH_SIZE,
            weights=weights, weights_grad=weights_grad,
            activation=ACTIVATION,
            optimizer=OPTIMIZER, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            alpha=ALPHA, beta_1=BETA_1, beta_2=BETA_2, momentum=MOMENTUM,
            scheduler=None, grad_clip=GRAD_CLIP, batch_norm=BATCH_NORM,
            early_stop=EARLY_STOP,
            critics_mode=CRITICS_MODE,
            lr_crit=LEARNING_RATE_CRIT,
            weight_decay_crit=WEIGHT_DECAY_CRIT,
            model_name=MODEL_NAME,
            path=PATH_MODELS
        )

    if TRAIN_NEW:
        print('Train model...')
        if START_FROM_EPOCH > 0:
            model, critics = M.load_model(
                model, critics,
                load_epoch=START_FROM_EPOCH, path=PATH_MODELS+output_model
            )

        print('Start training...')
        model, critics, loss_evol = T.train_turbo(
            model, opt_model,
            critics, opt_critics,
            data_x_train, data_z_train,
            data_x_valid, data_z_valid,
            mean_x, std_x, mean_z, std_z,
            weights,
            epochs=N_EPOCHS, batch_size=BATCH_SIZE,
            start_from=START_FROM_EPOCH,
            grad_clip=GRAD_CLIP,
            early_stop=EARLY_STOP,
            save=SAVE, path=PATH_MODELS, output_model=output_model,
            device=device
        )
        print('End training.')

        if SAVE:
            print('Save model...')
            torch.save(model.state_dict(),
                       PATH_MODELS + output_model + '_final.pt')
            for k in critics.keys():
                torch.save(
                    critics[k].state_dict(),
                    PATH_MODELS + output_model + f'_{k}_final.pt'
                )

        print('Reload best KS model...')
        model, critics = M.load_model(model, critics, which='best_ks',
                                      path=PATH_MODELS+output_model)
    else:
        print('Load model...')
        model, critics = M.load_model(model, critics, which=LOAD_WHICH,
                                      path=PATH_MODELS+output_model)

    if TRAIN_NEW:
        print('Plot losses...')
        P.plot_losses(loss_evol,
                      keys=[k for k, v in weights.items() if v != 0.],
                      n_epochs=N_EPOCHS, n_batch=n_batch,
                      save=SAVE, path=PATH_IMAGES+output_model)

    if PLOT:
        print('Plot 2D correlations...')
        P.plot_2D(
            model,
            data_x_test[:RECO_LIM], data_z_test[:RECO_LIM],
            mean_x, std_x, mean_z, std_z,
            show=False, save=SAVE, path=PATH_IMAGES
        )

        print('Plot data observables...')
        P.plot_hists(
            model,
            data_x_test, data_z_test,
            mean_x, std_x, mean_z, std_z,
            show=False, save=SAVE, path=PATH_IMAGES
        )

        print('Plot reco observables...')
        P.plot_hists_reco(
            model,
            data_x_test[:RECO_LIM], data_z_test[:RECO_LIM],
            mean_x, std_x, mean_z, std_z,
            show=False, save=SAVE, path=PATH_IMAGES
        )

    print("-> All done!")