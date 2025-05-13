
import torch
import pytest
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from model import *

def test_skip_connection():
    in_dim = 8
    out_dim = 16
    model = SkipConnection(in_dim, out_dim)
    x_old = torch.randn(4, in_dim)
    x_new = torch.randn(4, out_dim)
    output = model(x_old, x_new)
    assert output.shape == (4, out_dim)
    print("Skip connection test passed.")

def test_bond_embedding():
    model = BondEmbedding(6, 12)
    x = torch.randn(5, 6)
    out = model(x)
    assert out.shape == (5, 12)
    print("Bond embedding test passed.")

def test_graph_encoder():
    model = GraphEncoder(8, 16, 8, n_layers=2)
    x = torch.randn(10, 8)
    edge_index = torch.tensor([[0,1,2,1],[1,0,1,2]])
    edge_attr = torch.randn(4, 8)
    out = model(x, edge_index, edge_attr)
    final_dim = 8 + 2 * 16
    assert out.shape[0] == 10
    assert out.shape[1] == final_dim
    print("Graph encoder test passed.")

def test_atom_mapper():
    model = AtomMapper(8, 16, 8)
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0,1,2,1],[1,0,1,2]])
    edge_attr = torch.randn(4, 8)
    batch = torch.tensor([0,0,0,1,1,1])
    out = model(x, edge_index, edge_attr, batch, x, edge_index, edge_attr, batch)
    assert out.shape == (2, 3, 3)
    print("Atom mapper test passed.")

def test_atom_reactivity_predictor():
    model = AtomReactivityPredictor(16)
    x = torch.randn(10, 16)
    out = model(x)
    assert out.shape == (10,)
    print("Atom reactivity predictor test passed.")

def test_dual_graph_bond_reactivity_predictor():
    model = DualGraphBondReactivityPredictor(16)
    x = torch.randn(12, 16)
    out = model(x)
    assert out.shape == (12,)
    print("Dual graph bond reactivity predictor test passed.")

def test_bond_reactivity_predictor():
    model = BondReactivityPredictor(16, 8, 32, 32, 16)
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0,1,2,1],[1,0,1,2]])
    edge_attr = torch.randn(4, 8)
    dual_node_emb = torch.randn(4, 16)
    atom_logits = torch.randn(10)
    out = model(x, edge_index, edge_attr, dual_node_emb, atom_logits)
    assert out.shape == (4,)
    print("Bond reactivity predictor test passed.")

def test_guided_cross_attention():
    model = GuidedCrossAttention(16, n_heads=4)
    x = torch.randn(6, 16)
    y = torch.randn(6, 16)
    batch = torch.tensor([0,0,0,1,1,1])
    mapping = torch.tensor([0,1,2,0,1,2])
    out = model(x, y, batch, batch, mapping)
    assert out.shape[-1] == 16
    print("Guided cross attention test passed.")

def test_attention_rc_identifier_forward():
    model = ARC(8, 16, 6, 8, 6, n_heads=2, n_layers=2)
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0,1,2,1],[1,0,1,2]])    
    edge_attr = torch.randn(4, 6)
    batch = torch.tensor([0,0,0,1,1,1])
    mapping = torch.tensor([0,1,2,0,1,2])
    dual_edge_index = edge_index
    dual_edge_attr = edge_attr
    out = model(x, edge_index, edge_attr, batch, x, edge_index, edge_attr, batch,
                dual_edge_index, dual_edge_attr, mapping, use_ground_truth=True)
    soft_matching, atom_logits, bond_logits = out
    assert soft_matching.shape == (2, 3, 3)
    assert atom_logits.dim() == 1
    assert bond_logits.dim() == 1
    print("Attention RC identifier forward test passed.")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
