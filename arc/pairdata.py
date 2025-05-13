import torch
from torch_geometric.data import Data

class PairData(Data):
    def __init__(self, edge_index_r=None, x_r=None, edge_attr_r=None, n_r=None, map_r=None, b_r=None, z_r=None, batch_r=None,
                       edge_index_p=None, x_p=None, edge_attr_p=None, n_p=None, map_p=None, b_p=None, z_p=None, batch_p=None,
                       p2r_mapper = None, eq_as = None, rc_atoms = None,  rc_bonds = None, reaction_smiles = None, reaction_class = None):
        super().__init__()

        # Explicitly store attributes with default None handling
        self.edge_index_r = edge_index_r if edge_index_r is not None else torch.empty((2, 0), dtype=torch.long)
        self.x_r = x_r if x_r is not None else torch.empty((0, 0), dtype=torch.float)
        self.edge_index_p = edge_index_p if edge_index_p is not None else torch.empty((2, 0), dtype=torch.long)
        self.x_p = x_p if x_p is not None else torch.empty((0, 0), dtype=torch.float)
        self.edge_attr_r = edge_attr_r if edge_attr_r is not None else torch.empty((0, 0), dtype=torch.float)
        self.edge_attr_p = edge_attr_p if edge_attr_p is not None else torch.empty((0, 0), dtype=torch.float)
        self.map_r = map_r if map_r is not None else torch.empty((0,), dtype=torch.long)
        self.map_p = map_p if map_p is not None else torch.empty((0,), dtype=torch.long)
        self.n_r = n_r
        self.n_p = n_p
        self.b_r = b_r
        self.b_p = b_p
        self.z_r = z_r
        self.z_p = z_p
        self.p2r_mapper = p2r_mapper
        self.eq_as = eq_as
        self.rc_atoms = rc_atoms
        self.rc_bonds = rc_bonds
        self.reaction_smiles = reaction_smiles
        self.reaction_class = reaction_class
        # Custom batch
        self.batch_r = batch_r if batch_r is not None else torch.zeros(self.x_r.size(0), dtype=torch.long)
        self.batch_p = batch_p if batch_p is not None else torch.zeros(self.x_p.size(0), dtype=torch.long)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_r":
            return self.x_r.size(0) if self.x_r is not None else 0
        if key == "edge_index_p":
            return self.x_p.size(0) if self.x_p is not None else 0
        if key == "batch_r" or key == "batch_p":
            return 1  # batch incrementing
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["edge_index_r", "edge_index_p"]:
            return 1  
        if key in ["batch_r", "batch_p"]:
            return 0  
        return super().__cat_dim__(key, value, *args, **kwargs)

if __name__ == "__main__":
    sample = [
        PairData(
            edge_index_p=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_attr_p=torch.tensor([[1.0], [2.0]]),
            x_p=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            edge_index_r=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_attr_r=torch.tensor([[3.0], [4.0]]),
            x_r=torch.tensor([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
            map_r=torch.tensor([1, 2, 3]), 
            b_r=torch.tensor([1, 2, 3]),
            reaction_class= torch.tensor([1])
        ) for _ in range(8)  
    ]

    from torch_geometric.loader import DataLoader
    dataloader = DataLoader(sample, batch_size=5, shuffle=False)
    data = next(iter(dataloader))
    print(data)
    print("Batch assignment vector (data.batch_r):", data.batch_r)
    print("Batch assignment vector (data.batch_p):", data.batch_p)
