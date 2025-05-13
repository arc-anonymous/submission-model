import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch
from utils import *


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_attr_dim, n_layers=3, cat=True, learning_skip_connection=True):
        super(GraphEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cat = cat
        self.gnn_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        in_channels = input_dim
        out_channels = hidden_dim
        self.learning_skip_connection = learning_skip_connection

        for _ in range(n_layers):
            self.gnn_layers.append(GINEConv(nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),), edge_dim=edge_attr_dim))
            if self.learning_skip_connection:
                self.skip_connections.append(SkipConnection(in_channels, out_channels))
            else:
                if in_channels != out_channels:
                    self.skip_connections.append(nn.Linear(in_channels, out_channels))
                else:
                    self.skip_connections.append(nn.Identity())
            in_channels = out_channels
                
    def forward(self, x, edge_index, edge_attr):
        xs = [x]  
        if self.learning_skip_connection:
            for conv, skip in zip(self.gnn_layers, self.skip_connections):
                x_new = conv(xs[-1], edge_index, edge_attr=edge_attr)
                x = skip(xs[-1], x_new)  
                xs.append(x)
        else:
            for conv, skip in zip(self.gnn_layers, self.skip_connections):
                x_res = skip(xs[-1])  
                x_new = conv(xs[-1], edge_index, edge_attr=edge_attr)
                x = x_new + x_res  
                xs.append(x)
        node_embeddings = torch.cat(xs, dim=-1) if self.cat else xs[-1] 
        return node_embeddings

class AtomMapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_attr_dim, n_layers=3, cat=True, learning_skip_connection=True):
        super(AtomMapper, self).__init__()
        self.gnn = GraphEncoder(input_dim, hidden_dim, edge_attr_dim, n_layers, cat, learning_skip_connection)

    def forward(self, x_p, edge_index_p, edge_feat_p, batch_p, x_r, edge_index_r, edge_feat_r, batch_r):
        h_r = self.gnn(x_r, edge_index_r, edge_feat_r)
        h_p = self.gnn(x_p, edge_index_p, edge_feat_p)
        h_r, mask_r = to_dense_batch(h_r, batch_r, fill_value=0)
        h_p, mask_p = to_dense_batch(h_p, batch_p, fill_value=0)
        assert h_p.size(0) == h_r.size(0), 'Encountered unequal batch-sizes'
        B, N_p, embedding_dim = h_p.size()
        N_r = h_r.size(1)
        m_hat = h_p @ h_r.transpose(-1, -2)  # [B, N_p, N_r]
        valid_mask = mask_p.view(B, N_p, 1) & mask_r.view(B, 1, N_r)  # [B, N_p, N_r]
        similarity_matrix = masked_softmax(m_hat, valid_mask, dim=-1)  # [B, N_p, N_r]
        soft_matching = sinkhorn(similarity_matrix)  # [B, N_p, N_r]
        return soft_matching


class SkipConnection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SkipConnection, self).__init__()
        self.transform = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.alpha = nn.Parameter(torch.ones(1))  

    def forward(self, x_old, x_new):
        return self.alpha * self.transform(x_old) + (1 - self.alpha) * x_new
    
class BondEmbedding(nn.Module):
    def __init__(self, bond_input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(bond_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1))
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1))
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1))

    def forward(self, bond_attr):
        x = self.fc1(bond_attr)
        x = self.fc2(x)
        return self.fc3(x)

class AtomReactivityPredictor(nn.Module):
    def __init__(self, node_emb_dim, hidden_dim=512):
        super(AtomReactivityPredictor, self).__init__()
        self.node_emb_dim = node_emb_dim    
        # Atom reactivity classifier
        self.fc1 = nn.Linear(self.node_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) 

    def forward(self, node_embedding):
        x = F.silu(self.fc1(node_embedding))
        x = F.silu(self.fc2(x))
        atom_reactivity_logits = self.fc3(x).squeeze(-1) 
        return atom_reactivity_logits

class DualGraphBondReactivityPredictor(nn.Module):
    def __init__(self, dual_node_emb_dim, hidden_dim=512):
        super(DualGraphBondReactivityPredictor, self).__init__()
        self.fc1 = nn.Linear(dual_node_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bond_reactivity = nn.Linear(hidden_dim, 1) # Bond binary classification (is changed or not)

    def forward(self, dual_node_emb):
        x = F.silu(self.fc1(dual_node_emb))
        x = F.silu(self.fc2(x))
        bond_logits = self.bond_reactivity(x).squeeze(-1)
        return bond_logits

class BondReactivityPredictor(nn.Module):
    def __init__(self, node_emb_dim, edge_attr_dim, edge_hidden_dim, hidden_dim, dual_input_dim):
        super(BondReactivityPredictor, self).__init__()
        self.bond_embedding = BondEmbedding(edge_attr_dim, edge_hidden_dim)
        self.dual_graph_bond_reactivity_predictor = DualGraphBondReactivityPredictor( dual_input_dim, hidden_dim) #
        self.fc1 = nn.Linear(2 * node_emb_dim + edge_hidden_dim + 3,  hidden_dim) # 2 * node_emb_dim + edge_attr_dim + 2 attended atom embeddings + dual atom reactivity logits + dual bond reactivity logits
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bond = nn.Linear(hidden_dim, 1) # Bond binary classification (is changed or not)

    def forward(self, node_embedding, edge_index, edge_attr, dual_node_emb, atom_reactivity_logits= None):
        src, dst = edge_index
        edge_attr = self.bond_embedding(edge_attr)
        atom_reactivity_probs = torch.sigmoid(atom_reactivity_logits)  
        atom_reactivity_src = atom_reactivity_probs[src].unsqueeze(-1) 
        atom_reactivity_dst = atom_reactivity_probs[dst].unsqueeze(-1)
        dual_bond_reactivity_logits = self.dual_graph_bond_reactivity_predictor(dual_node_emb) 
        dual_bond_reactivity_probs = torch.sigmoid(dual_bond_reactivity_logits)  
        bond_features = torch.cat([node_embedding[src], node_embedding[dst],\
                                    edge_attr, \
                                    dual_bond_reactivity_probs.unsqueeze(-1), \
                                    atom_reactivity_src, atom_reactivity_dst
                                    ], dim=-1)
        x = F.silu(self.fc1(bond_features))
        x = F.silu(self.fc2(x))
        bond_reactivity_logits = self.bond(x).squeeze(-1)
        return bond_reactivity_logits


class GuidedCrossAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.heads = n_heads
        self.dim = dim
        self.scale = (dim // n_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim) #[B, N_p, D]
        self.k_proj = nn.Linear(dim, dim) #[B, N_r, D]
        self.v_proj = nn.Linear(dim, dim) #[B, N_r, D]

    def forward(self, h_p, h_r, batch_p, batch_r, mapping_matrix):
        """
        Each product atom (query) attends to mapped reactant atoms (keys).
        mapping_matrix: List or tensor of shape [B, N_p], where each row contains 
                        indices specifying which reactant atom corresponds to each 
                        product atom.
        """
        h_p_dense, mask_p = to_dense_batch(h_p, batch_p)  # [B, N_p_max, D]
        h_r_dense, mask_r = to_dense_batch(h_r, batch_r)  # [B, N_r_max, D]
        B, N_p, D = h_p_dense.size()
        N_r = h_r_dense.size(1)
        mapping_matrix_dense, _ = to_dense_batch(mapping_matrix, batch_p)  # [B, N_p_max]
        max_idx = h_r_dense.size(1) - 1
        mapping_matrix = torch.clamp(mapping_matrix_dense, 0, max_idx)
        assert mapping_matrix.size() == (B, N_p), 'Mapping matrix should have shape [B, N_p]'

        h_r_mapped = torch.gather(h_r_dense, dim=1, index=mapping_matrix.unsqueeze(-1).expand(-1, -1, D))  # [B, N_p, D]
        q = self.q_proj(h_p_dense).view(B, N_p, self.heads, D//self.heads).permute(0,2,1,3)  # [B, H, N_p, D/H] The product nodes are querying (looking for relevant reactant nodes).
        k = self.k_proj(h_r_mapped).view(B, N_p, self.heads, D//self.heads).permute(0,2,1,3)  # [B, H, N_p, D/H] The reactant nodes provide the keys (used to match the queries).
        v = self.v_proj(h_r_mapped).view(B, N_p, self.heads,  D//self.heads).permute(0,2,1,3)  # [B, H, N_p, D/H] The reactant nodes also provide the values (information to be retrieved).
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N_p, N_p]
        attn = attn.masked_fill(~mask_p.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(attn, dim=-1)
        out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(B, N_p, D)
        return out

class ARC(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_attr_dim, edge_hidden_dim, dual_edge_attr_dim, n_heads=2, n_layers=3, cat=True, learning_skip_connection=True):
        super(ARC, self).__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.cat = cat
        self.final_dim = (input_dim + n_layers * hidden_dim) if cat else hidden_dim
        bond_input_dim = (input_dim + n_layers * hidden_dim) if cat else hidden_dim
        self.dual_input_dim = edge_attr_dim
        self.dual_edge_attr_dim = dual_edge_attr_dim
        self.dual_final_dim = (self.dual_input_dim + n_layers * hidden_dim) if cat else hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.cross_attention = GuidedCrossAttention(dim=self.final_dim, n_heads=n_heads)
        D = self.final_dim
        self.fusion_mlp = nn.Sequential(nn.Linear(2*D, D), nn.ReLU(), nn.Linear(D, D))
        self.graph_encoder = GraphEncoder(self.input_dim, hidden_dim, edge_attr_dim,n_layers, self.cat, learning_skip_connection)
        self.dual_graph_endcoder= GraphEncoder(self.dual_input_dim, hidden_dim, dual_edge_attr_dim, n_layers, self.cat, learning_skip_connection)
        self.atom_mapper = AtomMapper(self.input_dim, hidden_dim, self.edge_attr_dim, n_layers, self.cat, learning_skip_connection)
        self.atom_reactivitys_predictor = AtomReactivityPredictor(self.final_dim)
        self.bond_reactivity_predictor = BondReactivityPredictor(bond_input_dim, self.edge_attr_dim, self.edge_hidden_dim ,hidden_dim, dual_input_dim=self.dual_final_dim)

    def forward(self, x_p, edge_index_p, edge_attr_p, batch_p, 
                x_r, edge_index_r, edge_attr_r, batch_r,
                 dual_edge_index_p, dual_edge_attr_p, 
                 mapping_matrix = None,use_ground_truth=True, symmetry_data=None):
        " During traing time, the mapping matrix is generated by the grounf truth mapping, and during inference time, it is generated by atom mapper module. "

        h_p = self.graph_encoder(x_p, edge_index_p, edge_attr_p) 
        h_r = self.graph_encoder(x_r, edge_index_r, edge_attr_r)
        dual_x = edge_attr_p
        dual_node_emb = self.dual_graph_endcoder(dual_x, dual_edge_index_p, dual_edge_attr_p )
        soft_matching = self.atom_mapper(x_p, edge_index_p, edge_attr_p, batch_p, x_r, edge_index_r, edge_attr_r, batch_r) 
        if use_ground_truth and mapping_matrix is not None:
            # Training mode: Use provided ground truth mapping
            mapping_matrix = mapping_matrix
        else:
            # Inference mode: Generate mapping matrix using atom mapper module
            mapping_matrix = get_symmetry_aware_atom_mapping(soft_matching, symmetry_data)
            
        attention_output = self.cross_attention(h_p, h_r, batch_p, batch_r, mapping_matrix)  # [B, N_p_max, D]        
        h_p_dense, mask = to_dense_batch(h_p, batch_p)  # [B, N_p_max, D]
        h_p_combined = torch.cat([h_p_dense, attention_output], dim=-1)  # [B, N_p_max, 2D]
        h_p_final = h_p_combined[mask]  # [num_nodes_p, 2D]
        h_p_final = self.fusion_mlp(h_p_final)
        atom_reactivity_logits = self.atom_reactivitys_predictor(h_p_final)
        bond_reactivity_logits = self.bond_reactivity_predictor(
            h_p_final, edge_index_p, edge_attr_p, dual_node_emb, atom_reactivity_logits)
        return soft_matching, atom_reactivity_logits, bond_reactivity_logits
