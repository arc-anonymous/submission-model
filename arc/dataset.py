import os
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Dataset
from rdkit import Chem
from pairdata import *
from utils_data import *

class ReactionDataset(Dataset):
    def __init__(self, root, split='train', num_wl_iterations=3, transform=None, pre_transform=None):
        """
        PyTorch Geometric Dataset for reaction data.
        Args:
            root (str): Directory where dataset is stored.
            split (str): Dataset split ('train', 'valid', 'test').
            num_wl_iterations (int): Number of Weisfeiler-Lehman iterations.
            transform: Transformations applied to data.
            pre_transform: Preprocessing transformations.
        """
        self.num_wl_iterations = num_wl_iterations
        self.split = split  # 'train', 'test', or 'valid'
        super().__init__(root, transform, pre_transform)
        self.data_list = self._load_data()

    def _load_data(self):
        processed_file = os.path.join(self.processed_dir, f"{self.split}.pt")
        if os.path.exists(processed_file):
            print(f"Loading processed data from {processed_file}...")
            return torch.load(processed_file)
        else:
            print(f"Processed file not found: {processed_file}. Processing data...")
            self.process()
            return torch.load(processed_file)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    @property
    def raw_file_names(self):
        return ["canonicalized_train.csv", "canonicalized_eval.csv", "canonicalized_test.csv"]

    @property
    def processed_file_names(self):
        return ["train.pt", "valid.pt", "test.pt"]

    def download(self):
        pass  # Implement download logic if needed

    def process(self):
        if self.split == 'test':
            raw_file = os.path.join(self.raw_dir, "canonicalized_test.csv")
            save_path = os.path.join(self.processed_dir, "test.pt")
        elif self.split == 'valid':
            raw_file = os.path.join(self.raw_dir, "canonicalized_eval.csv")
            save_path = os.path.join(self.processed_dir, "valid.pt")
        else:
            raw_file = os.path.join(self.raw_dir, "canonicalized_train.csv")
            save_path = os.path.join(self.processed_dir, "train.pt")
        if os.path.exists(save_path):
            print(f"Processed file already exists: {save_path}. Skipping processing.")
            return  # Skip processing if the file is already there

        print(f"Processing data for split: {self.split}")
        data_df = pd.read_csv(raw_file, names=['id', 'class', 'reaction'], skiprows=1)
        data_list = []
        for _, reactions in tqdm(data_df.iterrows(), total=data_df.shape[0]):
            reaction_smiles = reactions.reaction
            reactants_smiles, products_smiles = reaction_smiles.split('>>')

            reactants_mol = Chem.MolFromSmiles(reactants_smiles)
            products_mol = Chem.MolFromSmiles(products_smiles)

            x_r, edge_index_r, edge_attr_r = get_onehot_mol_features(reactants_mol)
            x_p, edge_index_p, edge_attr_p = get_onehot_mol_features(products_mol)

            n_r = torch.tensor(reactants_mol.GetNumAtoms(), dtype=torch.long)
            n_p = torch.tensor(products_mol.GetNumAtoms(), dtype=torch.long)

            z_r = torch.tensor([atom.GetAtomicNum() for atom in reactants_mol.GetAtoms()])
            z_p = torch.tensor([atom.GetAtomicNum() for atom in products_mol.GetAtoms()])

            b_r = torch.tensor([get_bond_type(bond) for bond in reactants_mol.GetBonds()])
            b_p = torch.tensor([get_bond_type(bond) for bond in products_mol.GetBonds()])

            map_r = torch.tensor(get_mapping_number(reactants_mol))
            map_p = torch.tensor(get_mapping_number(products_mol))

            eq_as = get_equivalent_atoms(reactants_mol, self.num_wl_iterations)
            p2r_mapper = torch.tensor(p2r_mapping(reaction_smiles), dtype=torch.long)

            full_reaction_center, bond_changes, hydrogen_changes, charge_changes = get_whole_reaction_center(reactants_mol, products_mol)

            atom_labels = torch.zeros(products_mol.GetNumAtoms(), dtype=torch.float)
            bond_labels = torch.zeros(products_mol.GetNumBonds(), dtype=torch.float)

            product_map2index = map2index(products_mol)
            bond_labels = [] #0 = no change, 1 = order change, 2 = breaking we exclude bond formations because it is rare in the dataset
            for src, dst in edge_index_p.t().tolist():
                if src in product_map2index and dst in product_map2index:
                    src_amap = product_map2index[src]
                    dst_amap = product_map2index[dst]
                    bond_key = tuple(sorted([src_amap, dst_amap]))
                    label = bond_changes.get(bond_key, 0)
                else:
                    label = 0 
                bond_labels.append(label)

            bond_labels = torch.tensor(bond_labels, dtype=torch.long)
            for atom_idx in full_reaction_center:
                atom_labels[product_map2index[atom_idx ]] = 1.0
            for atom_idx in hydrogen_changes:
                if atom_labels[atom_idx] == 0:  
                    atom_labels[atom_idx] = 1.0
            for atom_idx in charge_changes:
                if atom_labels[atom_idx] == 0:  
                    atom_labels[atom_idx] = 1.0
            data = PairData(
                edge_index_r=edge_index_r, x_r=x_r, edge_attr_r=edge_attr_r, n_r=n_r, b_r=b_r ,z_r=z_r, map_r=map_r,
                edge_index_p=edge_index_p, x_p=x_p, edge_attr_p=edge_attr_p, n_p=n_p, b_p=b_p, z_p=z_p, map_p=map_p,
                p2r_mapper = p2r_mapper, eq_as = eq_as, rc_atoms = atom_labels, rc_bonds = bond_labels, reaction_smiles = reaction_smiles, 
               reaction_class = reactions['class'])
            data_list.append(data)
        torch.save(data_list, save_path)
        print(f"Saved processed data to {save_path}")

if __name__ == "__main__":
    dataset_test = ReactionDataset(root='../datasets', split='test')
    dataset_valid = ReactionDataset(root='../datasets', split='valid')
    dataset_train = ReactionDataset(root='../datasets', split='train')
    print(len(dataset_train), len(dataset_valid), len(dataset_test))
    from torch_geometric.loader import DataLoader
    dataloader = DataLoader(dataset_test, batch_size=5, shuffle=False)
    data = next(iter(dataloader))
    print(dataset_test[10])
    print(dataset_test[8].p2r_mapper)
    print(dataset_test[8].rc_bonds)
    print(dataset_test[8].rc_bonds.shape)
    print(dataset_test[8].edge_index_p.shape)
    print(dataset_test[8].rc_atoms)
    print(data)
    print(data.batch_r)
    print(data.edge_attr_r.shape)
