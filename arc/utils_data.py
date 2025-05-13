import torch
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from pairdata import *
import os

def one_hot_encoding( x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def get_one_hot_atom_features( atom, use_chirality = True, hydrogens_implicit = True): 
    permitted_list_of_atoms = np.load('atom_list.npy')
    permitted_list_of_atoms = list(permitted_list_of_atoms) + ['NA']
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    atom_type  =  one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors  =  one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge  =  one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type  =  one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    num_hs  =  one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
    ex_valence =  one_hot_encoding(int(atom.GetExplicitValence()), list(range(1, 7)))
    imp_valence =  one_hot_encoding(int(atom.GetImplicitValence()), list(range(0, 6)))
    is_in_a_ring = [int(atom.IsInRing())]
    is_aromatic = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [round(float((atom.GetMass() - 10.812)/116.092),3)]
    vdw_radius_scaled = [round(float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6),3)] 
    covalent_radius_scaled = [round(float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76),3)]
    atom_feature_vector = atom_type +  n_heavy_neighbors + is_in_a_ring  + is_aromatic  + num_hs  \
                        + ex_valence + imp_valence  + atomic_mass_scaled \
                        + vdw_radius_scaled + covalent_radius_scaled  + hybridisation_type + formal_charge                            
    if use_chirality == True:
        chirality_type  =  one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type
    
    if hydrogens_implicit == True:
        n_hydrogens  =  one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens
    return np.array(atom_feature_vector)

def get_bond_type(bond):
    BOND_TYPE_TO_FLOAT = {
        None: 0.0,
        Chem.rdchem.BondType.SINGLE: 1.0,
        Chem.rdchem.BondType.DOUBLE: 2.0,
        Chem.rdchem.BondType.TRIPLE: 3.0,
        Chem.rdchem.BondType.AROMATIC: 1.5
    }
    return BOND_TYPE_TO_FLOAT[bond.GetBondType()]

def get_ring_size(bond):
    if bond.IsInRing():
        for ring_size in range(3, 9):  
            if bond.IsInRingSize(ring_size):
                return ring_size
    return 0

def get_one_hot_bond_features( bond, use_stereochemistry = True):
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type  =  one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj  = [int(bond.GetIsConjugated())]
    bond_is_in_ring  = [int(bond.IsInRing())]
    bond_is_aromatic  = [int(bond.GetIsAromatic())]
    bond_ring_size = one_hot_encoding(get_ring_size(bond), list(range(0, 9)))
    bond_feature_vector = bond_type  + bond_is_conj  + bond_is_in_ring + bond_is_aromatic + bond_ring_size 
    if use_stereochemistry == True:
        stereo_type  =  one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type
    return np.array(bond_feature_vector)

def get_onehot_mol_features(mol):
    n_nodes = mol.GetNumAtoms()
    silly_smiles = "O=O"
    silly_mol = Chem.MolFromSmiles(silly_smiles)
    node_feat_dim = len( get_one_hot_atom_features(silly_mol.GetAtomWithIdx(0)))
    edge_feat_dim = len( get_one_hot_bond_features(silly_mol.GetBondBetweenAtoms(0,1)))

    X = np.zeros((n_nodes, node_feat_dim))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] =  get_one_hot_atom_features(atom)  
    X = torch.tensor(X, dtype = torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        feat = get_one_hot_bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_feat_dim), dtype=torch.float)

    return X, edge_index, edge_attr

def get_mapping_number(mol):
    mapping = []
    for atom in mol.GetAtoms():
            mapping.append(atom.GetAtomMapNum()-1)
    return mapping


def idx2mapping( mol):
    idx2mapping= dict()
    for atom in mol.GetAtoms():
            idx2mapping[atom.GetIdx()] = atom.GetAtomMapNum()-1
    return idx2mapping

def p2r_mapping( reaction_smiles):
    reactantes_mol, product_mol =  get_reaction_mols(reaction_smiles)
    r_i2m, p_i2m=   idx2mapping(reactantes_mol),  idx2mapping(product_mol)
    mapping = dict()
    for k_r,v_r in r_i2m.items():
        for k_p, v_p in p_i2m.items():
            if v_r == v_p:
                mapping[k_p]=k_r
    
    sorted_dict = dict(sorted(mapping.items()))
    return  list(sorted_dict.values())


def get_reaction_mols(reaction_smiles):
    reactantes_smiles, products_smiles = reaction_smiles.split('>>')
    reactantes_mol = Chem.MolFromSmiles(reactantes_smiles)
    products_mol = Chem.MolFromSmiles(products_smiles)
    return reactantes_mol, products_mol 

def wl_atom_similarity( mol, num_wl_iterations):
    """
    Args:
        mol: RDKit molecule object.
        num_wl_iterations: Number of Weisfeiler-Lehman iterations to perform.
    Returns:
        dict: A dictionary atom indices to their updated labels.
    """
    label_dict = dict()
    for atom in mol.GetAtoms():
        label_dict[atom.GetIdx()]= atom.GetSymbol()
    for _ in range(num_wl_iterations):
        label_dict =  update_atom_labels(mol, label_dict)
    return label_dict

def update_atom_labels(mol, label_dict):
    """
    Updates atom labels based on sorted neighbor labels (WL iteration).
    """
    new_label_dict = {}
    for atom in mol.GetAtoms():
        neighbor_labels = sorted([label_dict[n.GetIdx()] for n in atom.GetNeighbors()])
        label_string = label_dict[atom.GetIdx()] + ''.join(neighbor_labels)
        new_label_dict[atom.GetIdx()] = label_string

    return new_label_dict


def get_equivalent_atoms( mol, num_wl_iterations):
    """
    Creates a list containing sets of equivalent atoms based on similarity in neighborhood.
    Args:
        mol: RDKit molecule object.
        num_wl_iterations: Number of Weisfeiler-Lehman iterations to perform.
    Returns:
        A list of sets where each set contains atom indices of equivalent atoms.
    """
    node_similarity =  wl_atom_similarity(mol, num_wl_iterations)
    n_h_dict = {atom.GetIdx(): atom.GetTotalNumHs() for atom in mol.GetAtoms()}
    degree_dict = {atom.GetIdx(): atom.GetDegree() for atom in mol.GetAtoms()}
    neighbor_dict = {atom.GetIdx(): [nbr.GetSymbol() for nbr in atom.GetNeighbors()] for atom in mol.GetAtoms()}
    
    atom_equiv_classes = []
    visited_atoms = set()
    
    for centralnode_indx, centralnodelabel in node_similarity.items():
        equivalence_class = set()
        
        if centralnode_indx not in visited_atoms:
            equivalence_class.add(centralnode_indx)
            visited_atoms.add(centralnode_indx)
        
        for firstneighbor_indx, firstneighborlabel in node_similarity.items():
            if (firstneighbor_indx not in visited_atoms and 
                centralnodelabel[0] == firstneighborlabel[0] and 
                set(centralnodelabel[1:]) == set(firstneighborlabel[1:]) and 
                degree_dict[centralnode_indx] == degree_dict[firstneighbor_indx] and 
                len(centralnodelabel) == len(firstneighborlabel) and 
                set(neighbor_dict[centralnode_indx]) == set(neighbor_dict[firstneighbor_indx]) and 
                n_h_dict[centralnode_indx] == n_h_dict[firstneighbor_indx]):
                
                equivalence_class.add(firstneighbor_indx)
                visited_atoms.add(firstneighbor_indx)
        
        if equivalence_class:
            atom_equiv_classes.append(equivalence_class)
    
    return atom_equiv_classes


def kekulize_mol(mol):
    mol = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol

def map2index(mol):
    return {atom.GetAtomMapNum() - 1: atom.GetIdx() for atom in mol.GetAtoms() if not atom.GetAtomMapNum() < 0}

def bond_info_amnum2btype(mol):
    if mol is None:
        return {}
    bond_info = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        amap1, amap2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
        if amap1 == 0 or amap2 == 0:
            continue
        bond_key = tuple(sorted([amap1 - 1, amap2 - 1]))
        bond_info[bond_key] = (bond.GetBondTypeAsDouble(), bond.GetIdx())
    return bond_info


def get_atom_numhorcharge_changes(reactant_mol, product_mol):
    if reactant_mol is None or product_mol is None:
        return set(), set()
    hydrogen_changes = set()
    charge_changes = set()
    reactant_map2index = map2index(reactant_mol)
    reactant_mol = Chem.AddHs(reactant_mol)
    product_mol = Chem.AddHs(product_mol)
    reactant_mol = kekulize_mol(reactant_mol)
    product_mol = kekulize_mol(product_mol)
    for atom in product_mol.GetAtoms():
        product_amnum = atom.GetAtomMapNum()
        if product_amnum == 0:  
            continue
        product_amnum -= 1 
        if product_amnum not in reactant_map2index:  
            continue
        prod_hydrogens = atom.GetTotalNumHs()
        prod_charge = atom.GetFormalCharge()
        reac_hydrogens = reactant_mol.GetAtomWithIdx(reactant_map2index[product_amnum]).GetTotalNumHs()
        react_charge = reactant_mol.GetAtomWithIdx(reactant_map2index[product_amnum]).GetFormalCharge()
        if prod_hydrogens != reac_hydrogens:
            hydrogen_changes.add(product_amnum)
        if prod_charge != react_charge:
            charge_changes.add(product_amnum)

    return hydrogen_changes, charge_changes 

def get_bond_changes_and_correspondance_atoms(reactant_mol, product_mol):
    """
    Identify atoms and bonds with changes, excluding bond formations (which don't exist in product).

    Returns:
    - atom_set: set of atom map numbers with bond changes
    - bond_edits: { (a1, a2): label } with labels:
        1 = bond order change
        2 = bond broken (exists in product but not in reactant)
    """
    reactant_mol = kekulize_mol(reactant_mol)
    product_mol = kekulize_mol(product_mol)
    atom_set = set()
    product_bonds = bond_info_amnum2btype(product_mol)
    reactant_bonds = bond_info_amnum2btype(reactant_mol)
    bond_edits = {}
    for bond, prod_bo in product_bonds.items():
        if bond in reactant_bonds:
            reac_bo = reactant_bonds[bond]
            if prod_bo[0] != reac_bo[0]:
                bond_edits[bond] = 1  # Bond order change
                atom_set.update(bond)
        else:
            bond_edits[bond] = 2  # Bond breaking
            atom_set.update(bond)

    return atom_set, bond_edits


def get_whole_reaction_center(reactant_mol, product_mol):
    atom_changed_bond, bond_changes = get_bond_changes_and_correspondance_atoms(reactant_mol, product_mol)
    hydrogen_changes, charge_changes = get_atom_numhorcharge_changes(reactant_mol, product_mol)
    full_reaction_center = atom_changed_bond.union(hydrogen_changes).union(charge_changes)
    return full_reaction_center, bond_changes, hydrogen_changes, charge_changes


def compute_graph_edit_distance(reactant_mol, product_mol):
    rcatoms, rcbonds, hydrogen_changes, charge_changes = get_whole_reaction_center(reactant_mol, product_mol)
    bond_edits = len(rcbonds)
    independent_hydrogen_edits = 0
    for atom in hydrogen_changes:
        is_bond_related = any(atom in bond for bond in rcbonds.keys())
        if not is_bond_related:
            independent_hydrogen_edits += 1
    
    charge_edits = len(charge_changes)
    total_edits = bond_edits + independent_hydrogen_edits + charge_edits
    return total_edits, bond_edits, independent_hydrogen_edits, charge_edits


# Function to create dual nodes from edge_index and edge_att
def create_dual_nodes(edge_index, edge_attr):
    edge_tuples = [tuple(sorted(e)) for e in edge_index.t().tolist()]
    edge_array = np.array(edge_tuples)
    unique_edges_np, indices_np = np.unique(edge_array, axis=0, return_index=True)
    unique_edges = torch.tensor(unique_edges_np)
    indices = torch.tensor(indices_np)
    dual_node_features = edge_attr[indices]
    return dual_node_features

def generate_dual_edges(edge_index):
    bond_dict = {}
    bond_connections = set()
    edges = {tuple(sorted(edge)) for edge in edge_index.t().tolist()}
    unique_edge_index = torch.tensor(list(edges)).T
    for bond_id, (atom1, atom2) in enumerate(unique_edge_index.t().tolist()):
        bond_dict.setdefault(atom1, []).append(bond_id)
        bond_dict.setdefault(atom2, []).append(bond_id)
    for bonds in bond_dict.values():
        if len(bonds) > 1:  
            for i in range(len(bonds)):
                for j in range(i + 1, len(bonds)):
                    bond_connections.add((bonds[i], bonds[j]))
    if bond_connections:
        dual_edges = torch.tensor(list(bond_connections), dtype=torch.long).t()
        return to_undirected(dual_edges)
    else:
        return torch.empty((2, 0), dtype=torch.long)  

def create_dual_edges(edge_index, x):
    dual_edge_index = []
    dual_edge_attr = []
    edges = {tuple(sorted(edge)) for edge in edge_index.t().tolist()}
    unique_edge_index = torch.tensor(list(edges)).T
    neighbors_dict = {}
    for edge in unique_edge_index.t().tolist():
        atom1, atom2 = edge
        if atom1 not in neighbors_dict:
            neighbors_dict[atom1] = []
        neighbors_dict[atom1].append(atom2)
        if atom2 not in neighbors_dict:
            neighbors_dict[atom2] = []
        neighbors_dict[atom2].append(atom1)
    for atom, bonds in neighbors_dict.items():
        for i in range(len(bonds)):
            for j in range(i+1, len(bonds)):
                b1, b2 = bonds[i], bonds[j]  
                dual_edge_index.append([b1, b2])  
                dual_edge_index.append([b2, b1])
                shared_atom_feat = x[atom]
                dual_edge_attr.extend([shared_atom_feat, shared_atom_feat])
    dual_edge_index = torch.tensor(dual_edge_index).t()  
    dual_edge_attr = torch.stack(dual_edge_attr)  
    return dual_edge_index, dual_edge_attr


def build_dual_graph(x, edge_index, edge_attr, rc_bonds= None):
    dual_node_features = create_dual_nodes( edge_index, edge_attr)
    dual_edge_index, dual_edge_attr = create_dual_edges( edge_index, x)
    dual_node_labels = rc_bonds
    dual_data = Data(
        x=dual_node_features,          
        edge_index=dual_edge_index,    
        edge_attr=dual_edge_attr,       
        y=dual_node_labels             
    )
    return dual_data

def get_dual_pairdata(data):
    dual_graph_r = build_dual_graph(data.x_r, data.edge_index_r, data.edge_attr_r)
    dual_graph_p = build_dual_graph(data.x_p, data.edge_index_p, data.edge_attr_p, data.rc_bonds)
    
    dual_data = PairData(
        x_p=dual_graph_p.x,
        edge_index_p=dual_graph_p.edge_index,
        edge_attr_p=dual_graph_p.edge_attr,
        x_r=dual_graph_r.x,
        edge_index_r=dual_graph_r.edge_index,
        edge_attr_r=dual_graph_r.edge_attr,
        reaction_class=data.reaction_class,
        rc_atoms=data.rc_bonds,
        reaction_smiles=data.reaction_smiles,
    )
    return dual_data
