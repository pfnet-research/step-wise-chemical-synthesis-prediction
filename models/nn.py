import chainer
from chainer import functions, reporter
from chainer_chemistry.links import GraphMLP
from models.gwm_custom import GWMGraphConvModel
from chainer_chemistry.links.update.ggnn_update import GGNNUpdate

import cupy as cp

class ggnngwm_stop_step(chainer.Chain):
    def __init__(self, out_dim, hidden_dim,
                 n_layers, concat_hidden,
                 weight_tying,
                 nn_hidden_dim,
                 gwm):
        super(ggnngwm_stop_step, self).__init__()
        with self.init_scope():
            self.ggnngwm = GWMGraphConvModel(out_dim=out_dim, hidden_channels=hidden_dim, n_update_layers=n_layers,
                                             update_layer=GGNNUpdate,
                                             concat_hidden=concat_hidden, weight_tying=weight_tying,
                                             with_gwm=gwm,
                                             n_edge_types=5)
            # output: binary classification after softmax
            self.mlp = GraphMLP(channels=[nn_hidden_dim, 2])

    def __call__(self, f_atoms, f_bonds, super_node_x, stop_idx, action, batch_size):

        loss, acc = 0.0, 0.0
        for step in range(action.shape[1]):
            action_step_ind = action[:, step, 0]

            h = self.ggnngwm(f_atoms, f_bonds, super_node_x)
            h = self.mlp(h[cp.arange(batch_size), stop_idx[:, 0]].reshape(batch_size, 1, h.shape[2]))[:, 0, :]
            l = (action_step_ind == -2).astype('int32') - (action_step_ind == -1).astype('int32')
            loss += functions.softmax_cross_entropy(h, l, ignore_label=-1)
            acc += functions.accuracy(h, l, ignore_label=-1)

            # execute one action in random order during each epoch
            action_step = action[:, step, :]
            for i in range(batch_size):
                a = action_step[i]
                if a[0] < 0:
                    continue
                else:
                    f_bonds[i, :4, a[0], a[1]] = 0.0
                    f_bonds[i, :4, a[1], a[0]] = 0.0
                    f_bonds[i, 4, a[0], a[1]] = 1.0
                    f_bonds[i, 4, a[1], a[0]] = 1.0
                    if a[2] + 1 != 0:
                        f_bonds[i, a[2], a[0], a[1]] = 1.0
                        f_bonds[i, a[2], a[1], a[0]] = 1.0

        return loss, acc / action.shape[1]


class ggnngwm_atom(chainer.Chain):
    def __init__(self, out_dim, hidden_dim,
                 n_layers, concat_hidden,
                 weight_tying,
                 nn_hidden_dim,
                 gwm,
                 topK):
        super(ggnngwm_atom, self).__init__()
        with self.init_scope():
            self.ggnngwm = GWMGraphConvModel(out_dim=out_dim, hidden_channels=hidden_dim, n_update_layers=n_layers,
                                             update_layer=GGNNUpdate,
                                             concat_hidden=concat_hidden, weight_tying=weight_tying,
                                             with_gwm=gwm,
                                             n_edge_types=5)
            # output: one sigmoid score of each atom
            self.mlp = GraphMLP(channels=[nn_hidden_dim, 1])
            self.topK = topK

    def __call__(self, f_atoms, f_bonds, super_node_x, atom_label, mask_reagents, mask_reactants_reagents, batch_size):
        h = self.ggnngwm(f_atoms, f_bonds, super_node_x)
        h = self.mlp(h)[:, :, 0]
        loss = functions.sigmoid_cross_entropy(h, atom_label)
        h = functions.sigmoid(h).array
        acc = atom_selection_acc(h, atom_label, mask_reagents, self.topK, batch_size)
        atom_selected = atom_selection(h, mask_reactants_reagents, self.topK)

        return loss, acc, atom_selected


class ggnngwm_pair_step(chainer.Chain):
    def __init__(self, out_dim, hidden_dim,
                 n_layers, concat_hidden,
                 weight_tying,
                 nn_hidden_dim,
                 gwm,
                 topK):
        super(ggnngwm_pair_step, self).__init__()
        with self.init_scope():
            self.ggnngwm = GWMGraphConvModel(out_dim=out_dim, hidden_channels=hidden_dim, n_update_layers=n_layers,
                                             update_layer=GGNNUpdate,
                                             concat_hidden=concat_hidden, weight_tying=weight_tying,
                                             with_gwm=gwm,
                                             n_edge_types=5)
            # output: one softmax score of each pair
            # |pairs|: 10*9/2=45
            self.mlp = GraphMLP(channels=[nn_hidden_dim, 1])
            self.out_dim = out_dim
            self.topK = topK

    def __call__(self, f_atoms, f_bonds, super_node_x, action, pair_label, mask_pair_select, batch_size, atom_selected):
        loss = 0.0
        mask_selection = cp.ones((batch_size, self.topK, self.topK))
        mask_selection = cp.triu(mask_selection, 1)
        mask_selection = mask_selection.reshape(batch_size, self.topK * self.topK)
        for step in range(action.shape[1] - 1):
            h = self.ggnngwm(f_atoms, f_bonds, super_node_x)
            f = cp.zeros((batch_size, self.topK, self.out_dim)).astype('float32')
            for i in range(batch_size):
                f[i] = h[i][atom_selected[i]].array
            f = functions.broadcast_to(functions.expand_dims(f, axis=2),
                                       (batch_size, self.topK, self.topK, self.out_dim)) + \
                functions.broadcast_to(functions.expand_dims(f, axis=1),
                                       (batch_size, self.topK, self.topK, self.out_dim))
            f = self.mlp(f.reshape(batch_size, self.topK * self.topK, self.out_dim))[:, :, 0]
            l = pair_label[:, step, :, :].reshape(batch_size, self.topK * self.topK)
            l = l * mask_selection

            m = mask_pair_select.reshape(batch_size, self.topK * self.topK) * mask_selection
            loss += softmax_cross_entropy_with_mask(f, l, m, batch_size)

            # execute one action in random order during each epoch
            action_step = action[:, step, :]
            mask_selection[cp.arange(batch_size), action_step[:, -1]] = 0

            for i in range(batch_size):
                a = action_step[i]
                if a[0] < 0:
                    continue
                else:
                    f_bonds[i, :4, a[0], a[1]] = 0.0
                    f_bonds[i, :4, a[1], a[0]] = 0.0
                    f_bonds[i, 4, a[0], a[1]] = 1.0
                    f_bonds[i, 4, a[1], a[0]] = 1.0
                    if a[2] + 1 != 0:
                        f_bonds[i, a[2], a[0], a[1]] = 1.0
                        f_bonds[i, a[2], a[1], a[0]] = 1.0

        return loss


class ggnngwn_action_step(chainer.Chain):
    def __init__(self, out_dim, hidden_dim,
                 n_layers, concat_hidden,
                 weight_tying,
                 nn_hidden_dim,
                 gwm):
        super(ggnngwn_action_step, self).__init__()
        with self.init_scope():
            self.ggnngwm = GWMGraphConvModel(out_dim=out_dim, hidden_channels=hidden_dim, n_update_layers=n_layers,
                                             update_layer=GGNNUpdate,
                                             concat_hidden=concat_hidden, weight_tying=weight_tying,
                                             with_gwm=gwm,
                                             n_edge_types=5)
            # output: [remove, single, double, triple] softmax score of each action for one pair
            self.mlp = GraphMLP(channels=[nn_hidden_dim, 5 - 1])  # no aromatic
            self.out_dim = out_dim

    def __call__(self, f_atoms, f_bonds, super_node_x, action, batch_size):
        loss, acc = 0.0, 0.0

        action_label = (action[:, :, 2] + 2) * (action[:, :, 0] >= 0).astype('int32') - 1

        for step in range(action.shape[1] - 1):
            h = self.ggnngwm(f_atoms, f_bonds, super_node_x)
            action_step = action[:, step, :]
            h = h[cp.arange(batch_size), action_step[:, 0]] + h[cp.arange(batch_size), action_step[:, 1]]
            h = h.reshape(batch_size, 1, self.out_dim)
            h = self.mlp(h)[:, 0, :]
            l = action_label[:, step]

            loss += functions.softmax_cross_entropy(h, l, ignore_label=-1)
            acc += functions.accuracy(h, l, ignore_label=-1)

            # execute one action in random order during each epoch
            action_step = action[:, step, :]
            for i in range(batch_size):
                a = action_step[i]
                if a[0] < 0:
                    continue
                else:
                    f_bonds[i, :4, a[0], a[1]] = 0.0
                    f_bonds[i, :4, a[1], a[0]] = 0.0
                    f_bonds[i, 4, a[0], a[1]] = 1.0
                    f_bonds[i, 4, a[1], a[0]] = 1.0
                    if a[2] + 1 != 0:
                        f_bonds[i, a[2], a[0], a[1]] = 1.0
                        f_bonds[i, a[2], a[1], a[0]] = 1.0

        return loss, acc / (action.shape[1] - 1)


def softmax_cross_entropy_with_mask(f, l, m, batch_size):
    h = f + (m - 1.) * 1e10
    loss = - functions.sum(functions.log_softmax(h) * l) / batch_size
    return loss


def pair_selection(f, m, bs):
    h = f + (m - 1.) * 1e10
    k = cp.argmax(functions.softmax(h.array).array, axis=1)
    # m[cp.arange(bs), k] = 0.
    return k

'''
acc of pair selection need to be further extended
'''
def pair_selection_acc(p, a, bs):
    cor = 0.0
    p = (a != 0).astype('int32') * p
    for i in range(bs):
        t = True
        for j in p[i]:
            if j == 0:
                break
            if j not in a[i]:
                t = False
                break
        # for j in a[i]:
        #     if j == 0:
        #         break
        #     if j not in p[i]:
        #         t = False
        #         break
        if t:
            cor += 1.
    return cor / bs

'''
9.19 discussion: reagents should not be masked during selection
'''
def atom_selection_acc(h, l, m, topK, bs):
    pred = cp.argsort(h + m)[:, -topK:]
    correct = 0
    for i in range(bs):
        p_idx = pred[i]
        l_idx = cp.where(l[i] == 1)[0]
        for idx in l_idx:
            if idx not in p_idx:
                break
            if idx == l_idx[-1]:
                correct += 1
    return correct / bs


def atom_selection(h, m, topK):
    pred = cp.argsort(h + m)[:, -topK:]
    return pred
