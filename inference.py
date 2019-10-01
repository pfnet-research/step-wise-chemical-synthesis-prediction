import chainer
from chainer import functions

import numpy as cp
# import cupy as cp

import json


def record(a, p, A, s, ind, d, name):
    for i in range(20):
        raw = d[ind[i]]
        dic = {
            'atoms': a[i].tolist(),
            'pairs': p[i].tolist(),
            'actions': A[i].tolist(),
            'stops': s[i].tolist(),
            'index': ind[i].tolist(),
            'raw': {
                'reactants_reagents': raw[0],
                'products': raw[1],
                'label': raw[2]
            }
        }
        with open(name + '_final.txt', 'a') as file:
            json.dump(dic, file)
            file.write('\n')


'''
whole framework: a->(p->A->s)
    
'''


def inference(g_stop, g_atom, g_pair, g_action,
              f_atoms, f_bonds, super_node_x,
              atom_label, mask_reagents, mask_reactants_reagents, pair_label, mask_pair_select,
              action, step_num,
              stop_idx,
              sample_index, data_raw, name):
    # select atoms
    h = g_atom.ggnngwm(f_atoms, f_bonds, super_node_x)
    h = g_atom.mlp(h)[:, :, 0]
    atoms_record = cp.argsort(functions.sigmoid(h).array + mask_reagents)[:, -10:]

    action_step_n = action.shape[1] - 1
    m = mask_pair_select.reshape(20, 10 * 10)
    pairs_record = cp.zeros((20, action_step_n, 2))
    stops_record = cp.zeros((20, action_step_n)).astype('float32')
    actions_record = cp.zeros((20, action_step_n)).astype('float32')

    # step-wise
    for step in range(action_step_n):

        # select one pair
        h = g_pair.ggnngwm(f_atoms, f_bonds, super_node_x)
        f = cp.zeros((20, 10, 100)).astype('float32')
        for i in range(20):
            f[i] = h[i][atoms_record[i]].array
        f = functions.broadcast_to(functions.expand_dims(f, axis=2),
                                   (20, 10, 10, 100)) + \
            functions.broadcast_to(functions.expand_dims(f, axis=1),
                                   (20, 10, 10, 100))
        f = g_pair.mlp(f.reshape(20, 10 * 10, 100))[:, :, 0]
        f = f + (m - 1.) * 1e10
        pairs_idx = cp.argmax(functions.softmax(f).array, axis=1)
        m[cp.arange(20), pairs_idx] = 0
        pairs_i = (pairs_idx / 10).astype('int32')
        pairs_j = (pairs_idx % 10).astype('int32')
        atoms_i = atoms_record[cp.arange(20), pairs_i]
        atoms_j = atoms_record[cp.arange(20), pairs_j]
        pairs_record[:, step, 0] = atoms_i
        pairs_record[:, step, 1] = atoms_j

        # predict the pair of that pair
        h = g_action.ggnngwm(f_atoms, f_bonds, super_node_x)
        h = h[cp.arange(20), atoms_i] + h[cp.arange(20), atoms_j]
        h = h.reshape(20, 1, 100)
        h = g_action.mlp(h)[:, 0, :]
        actions_record[:, step] = cp.argmax(functions.softmax(h).array, axis=1)

        # exec the predicted action
        for i in range(20):
            if 1. in stops_record[i, :]:
                continue
            else:
                f_bonds[i, :4, atoms_i[i], atoms_j[i]] = 0.0
                f_bonds[i, :4, atoms_j[i], atoms_i[i]] = 0.0
                f_bonds[i, 4, atoms_i[i], atoms_j[i]] = 1.0
                f_bonds[i, 4, atoms_j[i], atoms_i[i]] = 1.0
                ac = actions_record[i, step]
                if ac != 0.:
                    f_bonds[i, int(ac - 1), atoms_i[i], atoms_j[i]] = 1.0
                    f_bonds[i, int(ac - 1), atoms_j[i], atoms_i[i]] = 1.0

        # predict stop signal using the updated graph
        h = g_stop.ggnngwm(f_atoms, f_bonds, super_node_x)
        h = g_stop.mlp(h[cp.arange(20), stop_idx[:, 0]].reshape(20, 1, h.shape[2]))[:, 0, :]
        stops_record[:, step] = cp.argmax(functions.softmax(h).array, axis=1)

    # record the result by batch
    record(atoms_record, pairs_record, actions_record, stops_record, sample_index, data_raw, name)


if __name__ == '__main__':
    import chainer
    from chainer import serializers
    from chainer.iterators import SerialIterator
    from chainer_chemistry.dataset.converters import concat_mols

    from dataset import uspto_dataset
    from models.nn import ggnngwm_stop_step, ggnngwm_atom, ggnngwm_pair_step, ggnngwn_action_step

    import logging
    import argparse
    from distutils.util import strtobool

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hdim', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--nn_hidden_dim', type=int, default=50)
    parser.add_argument('--concat_hidden', type=strtobool, default='false')
    parser.add_argument('--weight_tying', type=strtobool, default='false')
    parser.add_argument('--gwm', type=strtobool, default='true')
    parser.add_argument('--topK', type=int, default=10)

    parser.add_argument('--g_stop', default='inference/snapshot_stop')
    parser.add_argument('--g_atom', default='inference/snapshot_atom')
    parser.add_argument('--g_pair', default='inference/snapshot_pair')
    parser.add_argument('--g_action', default='inference/snapshot_action')

    parser.add_argument('--test_path', default='dataset/test.txt.proc')
    parser.add_argument('--out', default='result_all/inference1')

    args = parser.parse_args()

    g_stop = ggnngwm_stop_step(out_dim=args.hdim, hidden_dim=args.hdim, n_layers=args.n_layers,
                               concat_hidden=args.concat_hidden, weight_tying=args.weight_tying,
                               nn_hidden_dim=args.nn_hidden_dim,
                               gwm=args.gwm)
    chainer.serializers.load_npz(args.g_stop, g_stop)

    g_atom = ggnngwm_atom(out_dim=args.hdim, hidden_dim=args.hdim, n_layers=args.n_layers,
                          concat_hidden=args.concat_hidden, weight_tying=args.weight_tying,
                          nn_hidden_dim=args.nn_hidden_dim,
                          gwm=args.gwm,
                          topK=args.topK)
    chainer.serializers.load_npz(args.g_atom, g_atom)

    g_pair = ggnngwm_pair_step(out_dim=args.hdim, hidden_dim=args.hdim, n_layers=args.n_layers,
                               concat_hidden=args.concat_hidden, weight_tying=args.weight_tying,
                               nn_hidden_dim=args.nn_hidden_dim,
                               gwm=args.gwm,
                               topK=args.topK)
    chainer.serializers.load_npz(args.g_pair, g_pair)

    g_action = ggnngwn_action_step(out_dim=args.hdim, hidden_dim=args.hdim, n_layers=args.n_layers,
                                   concat_hidden=args.concat_hidden, weight_tying=args.weight_tying,
                                   nn_hidden_dim=args.nn_hidden_dim,
                                   gwm=args.gwm)
    chainer.serializers.load_npz(args.g_action, g_action)

    # chainer.cuda.get_device_from_id(0).use()
    # g_stop.to_gpu()

    valid_raw = uspto_dataset.read_data(args.test_path)
    valid_dataset = uspto_dataset.USPTO_dataset(valid_raw)
    valid_iter = SerialIterator(valid_dataset, 20, repeat=False, shuffle=False)

    one_part_acc = []
    for batch in valid_iter:
        # get one batch of test data
        f_atoms, f_bonds, super_node_x, \
        atom_label, mask_reagents, mask_reactants_reagents, pair_label, mask_pair_select, \
        action, step_num, \
        stop_idx, \
        sample_index = concat_mols(batch, device=-1)

        atom_label -= 1
        mask_reagents -= 2
        mask_reactants_reagents -= 2
        action -= 1

        with chainer.using_config('train', False):
            inference(g_stop, g_atom, g_pair, g_action,
                      f_atoms, f_bonds, super_node_x,
                      atom_label, mask_reagents, mask_reactants_reagents, pair_label, mask_pair_select,
                      action, step_num,
                      stop_idx,
                      sample_index, valid_raw, args.out)
