"""
Utils for extract substructure associated to activated atoms.
An activated atom is one that meets the threshold (e.g. 0).
An activated bond is one that joins two activated atoms.
We find connected components specified by activated bonds,
and extract that connected component as substructure.
NB: This excludes activated atoms that are singletons,
IE, that don't share bonds with other activated atoms
"""
from rdkit import Chem
from rdkit.Chem import Draw
from plot_utils import plot_image_grid


def get_act_atoms(mol, weights, thres):
    return [i for i,a in enumerate(mol.GetAtoms()) if weights[i] > thres]


def get_bond_sets(mol, atom_inds):
    bond_sets = []
    for b in mol.GetBonds():
        if (b.GetBeginAtomIdx() in atom_inds and b.GetEndAtomIdx() in atom_inds):
            bond_sets.append(set((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
    return bond_sets


def find_ccs(unmerged):
    """
    Find connected components of a list of sets.
        E.g.
    x = [{'a','b'}, {'a','c'}, {'d'}]
    find_cc(x)
    
        [{'a','b','c'}, {'d'}]
    """
    merged = set()
    while unmerged:
        elem = unmerged.pop()
        shares_elements = False
        for s in merged.copy():
            if not elem.isdisjoint(s):
                merged.remove(s)
                merged.add(frozenset(s.union(elem)))
                shares_elements = True
        if not shares_elements:
            merged.add(frozenset(elem))

    return [list(x) for x in merged]


def atoms_to_bonds(mol, atom_inds):
    """
    Extract bonds indexes corresponding to a list of atom indexes
    """
    bond_inds = []
    for b in mol.GetBonds():
        if (b.GetBeginAtomIdx() in atom_inds and b.GetEndAtomIdx() in atom_inds):
            bond_inds.append(b.GetIdx())
    return bond_inds


def generate_active_substructs(data, thresh=0.0):
    """
    Generate list of active substructures
    """
    act_submols = []
    for d in data:
        smile = d['smile']
        weights = d['weights']

        mol = Chem.MolFromSmiles(smile)

        #Get activated atom indexes
        act_atom_inds = get_act_atoms(mol, weights, thresh)

        #Get bonds sets associated to activated atoms, e.g. b={AtomIdx1, AtomIdx2}, 
        act_bond_sets = get_bond_sets(mol, act_atom_inds)

        #Get connect components of activated bonds
        ccs = find_ccs(act_bond_sets)

        #Convert ccs back to bond inds
        act_bond_inds = [atoms_to_bonds(mol, cc) for cc in ccs]

        #Create submol obj for each cc
        submols = [Chem.PathToSubmol(mol, abi) for abi in act_bond_inds]
        act_submols.extend(submols)
    return act_submols


def count_substructs(substructs):
    """
    Given a list of non-unique substructures,
    count the occurrence of each.
    """
    counts = dict()
    for s in substructs:
        matched = False
        for ss in counts.keys():
            if s.HasSubstructMatch(ss) and ss.HasSubstructMatch(s):
                counts[ss] += 1
                matched = True
        if not matched:
            counts[s] = 1
    return counts


def count_substructz(molecules, query_substructs):
    """
    Given a list of molecules, count the occurrence of a list of query 
    substructures in that set of smiles.
    """
    assert isinstance(molecules[0], Chem.Mol)
    #Count number of times that substructure occurs in dataset
    counts = {}
    for substruct in query_substructs:
        count = 0
        for mol in molecules:
            count += len(mol.GetSubstructMatches(substruct))
        counts[substruct] = count
    return counts


def extract_topk(counts, key=0, topk=25):
    sorted_counts = sorted([(k,v) for k,v in counts.items()], reverse=True, key=lambda x:x[1])
    return [sorted_counts[k][key] for k in range(topk)]


def plot_structs(mols, labels, dataset):
    #Plot
    ims_arr = [[Draw.MolToImage(k, kekulize=False, size=(100,100)) for k in mols]]
    #freqs = [str(v) for k,v in sorted_counts[:topk]]
    row_labels_left = [ (dataset, '')]
    plot_image_grid(ims_arr, row_labels_left=row_labels_left, c=5, 
                             row_labels_right=[], 
                             col_labels=labels, 
                             super_col_labels=[],
                             col_rotation=0
                        )