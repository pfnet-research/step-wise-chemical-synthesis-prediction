import chainer
from chainer import reporter

import cupy as cp


class FrameworkUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.g_stop, self.g_atom, self.g_pair, self.g_action = kwargs.pop('models')
        super(FrameworkUpdater, self).__init__(*args, **kwargs)

        # store selected atoms for training pair selection
        self.atoms_selected = None

    def loss_stop(self, model, f_atoms, f_bonds, super_node_x, stop_idx, action, batch_size):
        loss, acc = model(f_atoms, f_bonds, super_node_x, stop_idx, action, batch_size)

        reporter.report({
            'loss_stop': loss,
            'acc_stop': acc
        }, model)

        return loss

    def loss_atom(self, model, f_atoms, f_bonds, super_node_x, atom_label, mask_reagents, mask_reactants_reagents, batch_size):
        loss, acc, self.atoms_selected = model(f_atoms, f_bonds, super_node_x, atom_label, mask_reagents, mask_reactants_reagents, batch_size)

        reporter.report({
            'loss_atom': loss,
            'acc_atom': acc
        }, model)

        return loss

    def loss_pair(self, model, f_atoms, f_bonds, super_node_x, action, pair_label, mask_pair_select, batch_size, atoms_selected):
        loss = model(f_atoms, f_bonds, super_node_x, action, pair_label, mask_pair_select, batch_size, atoms_selected)

        reporter.report({
            'loss_pair': loss,
            # 'acc_pair': acc
        }, model)

        return loss

    def loss_action(self, model, f_atoms, f_bonds, super_node_x, action, batch_size):
        loss, acc = model(f_atoms, f_bonds, super_node_x, action, batch_size)

        reporter.report({
            'loss_action': loss,
            'acc_action': acc
        }, model)

        return loss

    def update_core(self):

        # get 4 parts optimizers
        opt_stop = self.get_optimizer('opt_stop')
        opt_atom = self.get_optimizer('opt_atom')
        opt_pair = self.get_optimizer('opt_pair')
        opt_action = self.get_optimizer('opt_action')

        # load batch
        batch = self.get_iterator('main').next()

        f_atoms, f_bonds, super_node_x, \
        atom_label, mask_reagents, mask_reactants_reagents, pair_label, mask_pair_select, \
        action, step_num, \
        stop_idx, \
        sample_index = self.converter(batch, device=self.device)

        atom_label -= 1
        mask_reagents -= 2
        mask_reactants_reagents -= 2
        action -= 1
        batch_size = action.shape[0]

        # shuffle action, each sample will shuffle its action(except the stop action and padding) during each epoch
        for i in range(batch_size):
            cp.random.shuffle(action[i][:step_num[i]])

        # get each parts loss for parameter updating; report loss and acc of each part
        # f_bonds will be changed in each part, thus must use cp.copy
        opt_stop.update(self.loss_stop, self.g_stop, f_atoms, cp.copy(f_bonds), super_node_x, stop_idx, action, batch_size)

        opt_atom.update(self.loss_atom, self.g_atom, f_atoms, cp.copy(f_bonds), super_node_x,atom_label, mask_reagents, mask_reactants_reagents, batch_size)

        opt_pair.update(self.loss_pair, self.g_pair, f_atoms, cp.copy(f_bonds), super_node_x, action, pair_label, mask_pair_select, batch_size, self.atoms_selected)

        opt_action.update(self.loss_action, self.g_action, f_atoms, cp.copy(f_bonds), super_node_x, action, batch_size)