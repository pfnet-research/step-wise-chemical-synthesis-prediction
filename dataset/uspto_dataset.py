'''
Modified from https://github.com/wengong-jin/nips17-rexgen/blob/master/USPTO/core-wln-global/mol_graph.py
'''

import chainer

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

from chainer_chemistry.dataset.preprocessors.gwm_preprocessor import GGNNGWMPreprocessor

rdl = RDLogger.logger()
rdl.setLevel(RDLogger.CRITICAL)
elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
             'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
             'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U',
             'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']


def read_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            r, action = line.strip('\r\n ').split()
            if len(r.split('>')) != 3 or r.split('>')[1] != '': raise ValueError('invalid line:', r)
            react = r.split('>')[0]
            product = r.split('>')[-1]
            data.append([react, product, action])
    return data


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
                    + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                    + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     0  # add the check changed dimension.
                     ], dtype=np.float32)


def T0_data(reaction, sample_index, idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1):
    '''
    data preprocessing
    :param reaction: [0]: reactants and reagents; [1]: products; [2]: actions(pair with two reacted atom number and changed bond type)
    :param sample_index: the index of the reaction in raw txt
    :param idxfunc: get the real index in matrix
    :return: f_atoms: atom feature matrix with stop node
             f_bonds: atom adj feature matrix with stop node
             super_node_x: gwm create a super node to share updated feature between several molecules
             label: one-hot vector of atoms participated in reaction
             mask_reagents: mask -1 in the position of reagents
             mask_reactants_reagents: mask -1 in the position of reagents and give high values of reacted atoms
             pair_label: sort the reacted atoms' indies, then create pair matrix label. size=|steps|*|reacted atoms|*|reacted atoms|
             mask_pair_select: for |atoms|-|reagents| < 10, give 0 mask for pair matrics
             action_final: size=(|steps|+1)*4; for each step: [idx1, idx2, (bond type), (pair index in pair matrix)]; the added one step if for stop signal
             step_num: |action_final| - 1
             stop_idx: index of stop node
             sample_index: the index of the reaction in raw txt
    '''

    mol = Chem.MolFromSmiles(reaction[0])
    n_atoms = mol.GetNumAtoms()

    atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
    f_atoms = np.zeros((n_atoms + 1, atom_fdim))
    for atom in mol.GetAtoms():
        f_atoms[idxfunc(atom)] = atom_features(atom)

    f_bonds = np.zeros(
        (4 + 1, n_atoms + 1, n_atoms + 1))

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        bond_f = bond_features(bond)
        f_bonds[:, a1, a2] = bond_f
        f_bonds[:, a2, a1] = bond_f

    super_node_x = GGNNGWMPreprocessor().get_input_features(mol)[2]

    # 13-19-1.0;13-7-0.0 --> b=[12,18,6] ---> b=[6,12,18]
    b = []
    for a in reaction[2].split(';'):
        b.append(int(a.split('-')[0]) - 1)
        b.append(int(a.split('-')[1]) - 1)
    b = list(set(b))
    b.sort()

    # one-hot vector of reacted atoms, add stop node; will -=1 after padding
    label = np.ones(n_atoms + 1).astype(np.int32)
    label[b] = 2

    # action array: note that it stacked [-1, -1, -1] for stop step; will -=1 after padding
    action = np.array(reaction[2].replace(';', '-').split('-')).astype('float32').astype('int32').reshape(-1, 3)
    step_num = np.array(action.shape[0])
    assert step_num == len(reaction[2].split(';'))
    # actions should be shuffled
    np.random.shuffle(action)
    action = np.vstack([action, np.zeros(3).astype('int32') - 1])

    # stop node idx
    stop_idx = np.array([n_atoms])

    '''
    9.19 discussion: reagents should not be masked
    '''
    # reagents mask when select atoms; note that this mask will not used when calculating loss; will -=2 after padding
    mask_reagents = np.ones(n_atoms + 1).astype('int32')
    mask_reagents += 1
    mask_reagents[-1] = 0
    c = []
    for molecular in reaction[0].split('.'):
        reactant_bool = False
        for atomIdx in b:
            if ':' + str(atomIdx + 1) + ']' in molecular:
                reactant_bool = True
                break
        if reactant_bool is False:
            m_tmp = Chem.MolFromSmiles(molecular)
            for atom_tmp in m_tmp.GetAtoms():
                c.append(idxfunc(atom_tmp))
    mask_reagents[c] = 1

    # reagents mask is same as mask_reagents, reactants mask give large values according to sorted b list; will -=2 after padding
    mask_reactants_reagents = np.ones(n_atoms + 1).astype('int32')
    mask_reactants_reagents += 1
    mask_reactants_reagents[-1] = 0
    mask_reactants_reagents[c] = 1
    for bb in range(len(b)):
        mask_reactants_reagents[b[bb]] = 3 + bb

    # step-wise pair label, because the action is random shuffeled in each epoch, thus pair label matrics have no -1 mask for selected pair
    pair_label = np.zeros((step_num, 10, 10)).astype('float32')
    for s in range(step_num):
        for k in range(step_num):
            p = 1 / (step_num - s)
            acidx1 = 10 - len(b) + np.where(b == (action[k, 0] - 1))[0]
            acidx2 = 10 - len(b) + np.where(b == (action[k, 1] - 1))[0]
            pair_label[s, acidx1, acidx2] = p
            pair_label[s, acidx2, acidx1] = p
    pair_label = np.triu(pair_label, 1)

    # extend action to |step+1|*4; in each step, the last value stands for the pair's index in pair matrics
    action_idx = np.zeros(step_num + 1).astype('int32')
    action_final = np.zeros((step_num + 1, 4)).astype('int32')
    for s in range(step_num):
        acidx1 = 10 - len(b) + np.where(b == (action[s, 0] - 1))[0]
        acidx2 = 10 - len(b) + np.where(b == (action[s, 1] - 1))[0]
        action_idx[s] = min(acidx1, acidx2) * 10 + max(acidx1, acidx2) + 1
    for s in range(step_num + 1):
        action_final[s] = np.append(action[s], action_idx[s])

    # if |atoms|-|reagents| < 10, give 0 mask for pair label matrics
    could_select_n = n_atoms - len(c)
    mask_pair_select = np.ones((10, 10)).astype('float32')
    if could_select_n < 10:
        mask_pair_select[:(10 - could_select_n), :] = 0
        mask_pair_select[:, :(10 - could_select_n)] = 0
    mask_pair_select = np.triu(mask_pair_select, 1)

    f_atoms = f_atoms.astype('float32')
    f_bonds = f_bonds.astype('float32')
    sample_index = np.array(sample_index).astype('int32')
    return f_atoms, f_bonds, super_node_x, \
           label, mask_reagents, mask_reactants_reagents, pair_label, mask_pair_select, \
           action_final, step_num, \
           stop_idx, \
           sample_index


class USPTO_dataset(chainer.dataset.DatasetMixin):
    def __init__(self, reaction_list):
        self.reaction_list = reaction_list
        self.T0_data_list = [None for _ in range(len(self.reaction_list))]

        # for i in tqdm(range(len(self.reaction_list))):
        #     self.T0_data_list.append(T0_data(self.reaction_list[i]))

    def __len__(self):
        return len(self.reaction_list)

    def get_example(self, i):
        # cache for parallel data processing
        if self.T0_data_list[i] is None:
            self.T0_data_list[i] = T0_data(self.reaction_list[i], sample_index=i)
        return self.T0_data_list[i]
