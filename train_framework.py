import chainer
from chainer import training
from chainer.iterators import SerialIterator
from chainer_chemistry.dataset.converters import concat_mols
from chainer.training import extensions, StandardUpdater

import chainermn

import random
import logging
import argparse
from distutils.util import strtobool

from dataset import uspto_dataset
from models.nn import ggnngwm_stop_step, ggnngwm_atom, ggnngwm_pair_step, ggnngwn_action_step
from models.updater import FrameworkUpdater
from models.evaluater import FrameworEvaluater

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pairlr', type=float, default=1e-3)
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--out', '-o', default='result_debug',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--decay_iter', type=int, default=40000)

    parser.add_argument('--gwm', type=strtobool, default='true')
    parser.add_argument('--hdim', type=int, default=100, help='hidden dim')
    parser.add_argument('--n_layers', type=int, default=1,  # 3, 1
                        help='number of layers of encoder, decoder')
    parser.add_argument('--concat_hidden', type=strtobool, default='false')
    parser.add_argument('--weight_tying', type=strtobool, default='true')
    parser.add_argument('--nn_hidden_dim', type=int, default=50)
    parser.add_argument('--topK', type=int, default=10)

    parser.add_argument('--train_path', default='dataset/train.txt.proc')
    parser.add_argument('--valid_path', default='dataset/test.txt.proc')
    parser.add_argument('--size', default='all')

    parser.add_argument('--communicator', type=str, default='pure_nccl',
                        help='Type of communicator')

    args = parser.parse_args()

    # data parallel
    if args.gpu:
        comm = chainermn.create_communicator(args.communicator)
        device = comm.intra_rank
    else:
        comm = chainermn.create_communicator('naive')
        device = -1

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num Layers: {}'.format(args.n_layers))
        print('Num Hidden-dim: {}'.format(args.hdim))
        print('Num Minibatch-size: {}'.format(args.batch_size))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    g_stop = ggnngwm_stop_step(out_dim=args.hdim, hidden_dim=args.hdim, n_layers=args.n_layers,
                               concat_hidden=args.concat_hidden, weight_tying=args.weight_tying,
                               nn_hidden_dim=args.nn_hidden_dim,
                               gwm=args.gwm)
    g_atom = ggnngwm_atom(out_dim=args.hdim, hidden_dim=args.hdim, n_layers=args.n_layers,
                          concat_hidden=args.concat_hidden, weight_tying=args.weight_tying,
                          nn_hidden_dim=args.nn_hidden_dim,
                          gwm=args.gwm,
                          topK=args.topK)

    g_pair = ggnngwm_pair_step(out_dim=args.hdim, hidden_dim=args.hdim, n_layers=args.n_layers,
                               concat_hidden=args.concat_hidden, weight_tying=args.weight_tying,
                               nn_hidden_dim=args.nn_hidden_dim,
                               gwm=args.gwm,
                               topK=args.topK)
    g_action = ggnngwn_action_step(out_dim=args.hdim, hidden_dim=args.hdim, n_layers=args.n_layers,
                                   concat_hidden=args.concat_hidden, weight_tying=args.weight_tying,
                                   nn_hidden_dim=args.nn_hidden_dim,
                                   gwm=args.gwm)

    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        g_stop.to_gpu()
        g_atom.to_gpu()
        g_pair.to_gpu()
        g_action.to_gpu()


    def make_optimizer(model, alpha=args.lr):
        opt = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(alpha=alpha), comm)
        opt.setup(model)
        return opt


    opt_atom = make_optimizer(g_atom)
    opt_stop = make_optimizer(g_stop)
    opt_pair = make_optimizer(g_pair, args.pairlr)
    opt_action = make_optimizer(g_action)

    train_raw = uspto_dataset.read_data(args.train_path)
    valid_raw = uspto_dataset.read_data(args.valid_path)

    if comm.rank == 0:
        if args.size == 'debug':
            train_dataset = uspto_dataset.USPTO_dataset(train_raw[:100])
            valid_dataset = uspto_dataset.USPTO_dataset(valid_raw[:40])
        elif args.size == 'normal':
            random.shuffle(train_raw)
            train_dataset = uspto_dataset.USPTO_dataset(train_raw[:15000])
            valid_dataset = uspto_dataset.USPTO_dataset(valid_raw)
        elif args.size == 'all':
            train_dataset = uspto_dataset.USPTO_dataset(train_raw)
            valid_dataset = uspto_dataset.USPTO_dataset(valid_raw)
    else:
        train_dataset, valid_dataset = None, None

    train_dataset = chainermn.scatter_dataset(train_dataset, comm, shuffle=True)
    valid_dataset = chainermn.scatter_dataset(valid_dataset, comm, shuffle=True)

    train_iter = SerialIterator(train_dataset, args.batch_size)
    valid_iter = SerialIterator(valid_dataset, args.batch_size, repeat=False, shuffle=False)

    updater = FrameworkUpdater(
        models=(g_stop, g_atom, g_pair, g_action),
        iterator=train_iter,
        optimizer={'opt_stop': opt_stop, 'opt_atom': opt_atom, 'opt_pair': opt_pair, 'opt_action': opt_action},
        device=device,
        converter=concat_mols
    )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    evaluator = FrameworEvaluater(g_stop, g_atom, g_pair, g_action)
    evaluator = extensions.Evaluator(valid_iter, evaluator, device=device, converter=concat_mols)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    trainer.extend(extensions.observe_value('opt_stop/alpha', lambda t: opt_stop.alpha))
    trainer.extend(extensions.ExponentialShift('alpha', 0.9, optimizer=opt_stop),
                   trigger=(args.decay_iter, 'iteration'))
    trainer.extend(extensions.observe_value('opt_atom/alpha', lambda t: opt_atom.alpha))
    trainer.extend(extensions.ExponentialShift('alpha', 0.9, optimizer=opt_atom),
                   trigger=(args.decay_iter, 'iteration'))
    trainer.extend(extensions.observe_value('opt_pair/alpha', lambda t: opt_pair.alpha))
    trainer.extend(extensions.ExponentialShift('alpha', 0.9, optimizer=opt_pair),
                   trigger=(args.decay_iter, 'iteration'))
    trainer.extend(extensions.observe_value('opt_action/alpha', lambda t: opt_action.alpha))
    trainer.extend(extensions.ExponentialShift('alpha', 0.9, optimizer=opt_action),
                   trigger=(args.decay_iter, 'iteration'))

    if comm.rank == 0:
        frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
        trainer.extend(extensions.snapshot_object(g_stop, 'stop_snapshot_{.updater.iteration}'),
                       trigger=(frequency, 'epoch'))
        trainer.extend(extensions.snapshot_object(g_atom, 'atom_snapshot_{.updater.iteration}'),
                       trigger=(frequency, 'epoch'))
        trainer.extend(extensions.snapshot_object(g_pair, 'pair_snapshot_{.updater.iteration}'),
                       trigger=(frequency, 'epoch'))
        trainer.extend(extensions.snapshot_object(g_action, 'action_snapshot_{.updater.iteration}'),
                       trigger=(frequency, 'epoch'))

        trainer.extend(extensions.LogReport())

        trainer.extend(extensions.PrintReport(
            ['epoch',
             'opt_stop/alpha', 'opt_pair/alpha', 'opt_atom/alpha', 'opt_action/alpha',
             'opt_stop/loss_stop', 'validation/main/loss_stop',
             'opt_atom/loss_atom', 'validation/main/loss_atom',
             'opt_pair/loss_pair', 'validation/main/loss_pair',
             'opt_action/loss_action', 'validation/main/loss_action',
             'opt_stop/acc_stop', 'validation/main/acc_stop',
             'opt_atom/acc_atom', 'validation/main/acc_atom',
             'opt_pair/acc_pair', 'validation/main/acc_pair',
             'opt_action/acc_action', 'validation/main/acc_action',
             'elapsed_time']))

        trainer.extend(extensions.PlotReport(
            [
                'opt_stop/loss_stop',
                'opt_atom/loss_atom',
                'opt_pair/loss_pair',
                'opt_action/loss_action',
            ],
            'epoch', file_name='train_loss.png'))
        trainer.extend(extensions.PlotReport(
            [
                'validation/main/loss_stop',
                'validation/main/loss_atom',
                'validation/main/loss_pair',
                'validation/main/loss_action',
            ],
            'epoch', file_name='valid_loss.png'))
        trainer.extend(extensions.PlotReport(
            [
                'opt_stop/acc_stop',
                'opt_atom/acc_atom',
                'opt_pair/acc_pair',
                'opt_action/acc_action',
            ],
            'epoch', file_name='train_acc.png'))
        trainer.extend(extensions.PlotReport(
            [
                'validation/main/acc_stop',
                'validation/main/acc_atom',
                'validation/main/acc_pair',
                'validation/main/acc_action',
            ],
            'epoch', file_name='valid_acc.png'))

        trainer.extend(extensions.ProgressBar())

    trainer.run()

#    serializers.save_npz(args.out + '/model.npz', model)
