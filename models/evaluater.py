import chainer
from chainer import functions, reporter

import cupy as cp

class FrameworEvaluater(chainer.Chain):
    '''
        evaluator for each separate part
    '''

    def __init__(self, g_stop, g_atom, g_pair, g_action):
        self.g_stop = g_stop
        self.g_atom = g_atom
        self.g_pair = g_pair
        self.g_action = g_action
        super(FrameworEvaluater, self).__init__()

    def __call__(self, f_atoms, f_bonds, super_node_x,
           atom_label, mask_reagents, mask_reactants_reagents, pair_label, mask_pair_select,
           action, step_num,
           stop_idx,
           sample_index):

        atom_label -= 1
        mask_reagents -= 2
        mask_reactants_reagents -= 2
        action -= 1
        batch_size = action.shape[0]

        # atom
        loss_atom, acc_atom, atoms_selected = self.g_atom(f_atoms, cp.copy(f_bonds), super_node_x, atom_label, mask_reagents, mask_reactants_reagents, batch_size)

        # pair
        loss_pair = self.g_pair(f_atoms, cp.copy(f_bonds), super_node_x, action, pair_label, mask_pair_select, batch_size, atoms_selected)

        # action
        loss_action, acc_action = self.g_action(f_atoms, cp.copy(f_bonds), super_node_x, action, batch_size)

        # stop
        loss_stop, acc_stop = self.g_stop(f_atoms, cp.copy(f_bonds), super_node_x, stop_idx, action, batch_size)


        reporter.report({
            'loss_stop': loss_stop,
            'loss_atom': loss_atom,
            'loss_pair': loss_pair,
            'loss_action': loss_action,
            'acc_stop': acc_stop,
            'acc_atom': acc_atom,
            # 'acc_pair': acc_pair,   # acc_pair need to be further extended
            'acc_action': acc_action,
        }, self)
