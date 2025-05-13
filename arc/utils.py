
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc, roc_curve

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def masked_softmax(cm, mask, dim=-1):
    out = cm.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out

def sinkhorn(similarity_matrix, n_iters=100, epsilon=1e-9):
    log_alpha = similarity_matrix + epsilon
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)
def match_nodes(h_p, h_r, data):
    h_r, mask_r = to_dense_batch(h_r, data.batch_r, fill_value=0)
    h_p, mask_p = to_dense_batch(h_p, data.batch_p, fill_value=0)
    assert h_p.size(0) == h_r.size(0), 'Encountered unequal batch-sizes'
    B, N_p, embedding_dim = h_p.size()
    N_r = h_r.size(1)
    m_hat = h_p @ h_r.transpose(-1, -2)  # [B, N_p, N_r]
    valid_mask = mask_p.view(B, N_p, 1) & mask_r.view(B, 1, N_r)  # [B, N_p, N_r]
    similarity_matrix = masked_softmax(m_hat, valid_mask, dim=-1)  # [B, N_p, N_r]
    soft_matching = sinkhorn(similarity_matrix)  # [B, N_p, N_r]
    return soft_matching

def compute_loss(soft_matching, data):

    ground_truth, mask_p = to_dense_batch(data.p2r_mapper, data.batch_p, fill_value=-1)  # [B, N_p]
    valid_mask = mask_p  # [B, N_p]
    soft_matching_flat = soft_matching.view(-1, soft_matching.size(-1))  # [B * N_p, N_r]
    ground_truth_flat = ground_truth.view(-1)  # [B * N_p]
    valid_mask_flat = valid_mask.view(-1)  # [B * N_p]
    soft_matching_flat = soft_matching_flat[valid_mask_flat]  # [num_valid_nodes, N_r]
    ground_truth_flat = ground_truth_flat[valid_mask_flat]  # [num_valid_nodes]
    loss = F.nll_loss(F.log_softmax(soft_matching_flat, dim=-1), ground_truth_flat, reduction='mean')
    return loss

def select_matched_nodes(soft_matching,top_k=1):
    if top_k == 1:
        return torch.argmax(soft_matching, dim=-1)
    else:
        top_k_matches = torch.topk(soft_matching, k=top_k, dim=-1).indices
        return top_k_matches[:, 0]

def calculate_accuracy(soft_matching, data):
    ground_truth, mask_p = to_dense_batch(data.p2r_mapper, data.batch_p, fill_value=-1)  # [B, N_p]
    valid_mask = mask_p  # [B, N_p]
    predicted_matches = soft_matching.argmax(dim=-1)  # [B, N_p]
    valid_mask = valid_mask  # [B, N_p]
    predicted_matches = predicted_matches[valid_mask]  # [num_valid_nodes]
    ground_truth = ground_truth[valid_mask]  # [num_valid_nodes]
    correct = (predicted_matches == ground_truth).sum().item()
    total = valid_mask.sum().item()
    accuracy = correct / total if total > 0 else 0.
    return accuracy


def calculate_top_k_accuracy(soft_matching, data, k=1):
    ground_truth, mask_p = to_dense_batch(data.p2r_mapper, data.batch_p, fill_value=-1)  # [B, N_p]
    valid_mask = mask_p  # [B, N_p]
    _, top_k_predicted = torch.topk(soft_matching, k=k, dim=-1)  # [B, N_p, k]
    valid_mask = valid_mask  # [B, N_p]
    top_k_predicted = top_k_predicted[valid_mask]  # [num_valid_nodes, k]
    ground_truth = ground_truth[valid_mask]  # [num_valid_nodes]
    correct = torch.any(top_k_predicted == ground_truth.unsqueeze(-1), dim=-1).sum().item()  # [num_valid_nodes]
    total = valid_mask.sum().item()
    top_k_accuracy = correct / total if total > 0 else 0.0
    return top_k_accuracy

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def convert_to_tuple(element):
    if isinstance(element, set):
        return tuple(convert_to_tuple(item) for item in element)
    elif isinstance(element, (list, tuple)):
        return type(element)(convert_to_tuple(item) for item in element)
    else:
        return element

def get_atom_to_set(data):
    flat_eq_as = flatten_list(data.eq_as)
    return {atom: convert_to_tuple(atom_set) for atom_set in flat_eq_as for atom in convert_to_tuple(atom_set)}


def get_symmetry_aware_atom_mapping(soft_matching, data):
    """
    Generates a symmetry-aware atom mapping by ensuring that equivalent atoms
    are assigned distinct but valid predictions.

    Args:
        soft_matching (torch.Tensor): The soft matching scores between the nodes of the two sets.
        data (Data): The data containing information about the atoms and their equivalence classes.

    Returns:
        torch.Tensor: A list of predicted atom mappings after considering symmetry.
    """
    pred = select_matched_nodes(soft_matching).tolist()
    pred = pred[:data.n_p]  
    if isinstance(pred[0], list):
        pred = pred[0]  
    atom_to_set = get_atom_to_set(data)
    assigned_atoms = {}
    globally_assigned_atoms = set()

    for i, pr in enumerate(pred):
        pr_key = pr[0] if isinstance(pr, list) else pr

        if pr_key in atom_to_set:
            atom_set = sorted(atom_to_set[pr_key])  
            set_id = tuple(atom_set) 

            if set_id not in assigned_atoms:
                assigned_atoms[set_id] = set()

            available_atoms = [a for a in atom_set if a not in globally_assigned_atoms]

            if available_atoms:
                pred[i] = available_atoms[0]
                assigned_atoms[set_id].add(pred[i])
            assigned_atoms[set_id].add(pred[i])
            globally_assigned_atoms.add(pred[i])

    return torch.tensor(pred)

def get_symmetry_aware_accuracy(symmetry_aware_mapping, data):
    ground_truth = data.p2r_mapper[:data.n_p]
    n = data.n_p.item()
    symmetry_aware_mapping_accuracy = torch.sum((symmetry_aware_mapping == ground_truth)).item() / n

    return symmetry_aware_mapping_accuracy

def compute_metrics(y_true, y_pred_probs, threshold, task_name="atom"):
    has_positive = len(np.unique(y_true)) > 1
    y_pred = (y_pred_probs >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    if has_positive:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
        pr_curve_precision, pr_curve_recall, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(pr_curve_recall, pr_curve_precision)
    else:
        roc_auc = 0.0
        pr_auc = 0.0
    return {
        f"{task_name}_f1": f1,
        f"{task_name}_auc": roc_auc,
        f"{task_name}_pr_auc": pr_auc,
        f"{task_name}_precision": precision,
        f"{task_name}_recall": recall,
    }

class FocalLoss(nn.Module):
    """
     Focal Loss for atom change prediction.
    This loss function is designed to address class imbalance by focusing more on hard-to-classify examples.
    Args:
        positive_weights (torch.Tensor, optional): Class weights for positive samples. Defaults to None.
        gamma (float, optional): Focusing parameter. Defaults to 2.
        reduction (str, optional): Reduction method. Options are 'none', 'mean', or 'sum'. Defaults to 'mean'.
    """
    def __init__(self, positive_weights=None, gamma=2, reduction='mean'):  
        super(FocalLoss, self).__init__()
        self.positive_weights = positive_weights  # Class weights
        self.gamma = gamma  # Focus factor, which controls how much the easy examples are down-weighted
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        """
        Computes the  focal loss between inputs and targets.
        Args:
            inputs (Tensor): Predicted logits () of shape (N,).
            targets (Tensor): Ground truth labels () of shape (N,).
        Returns:
            Tensor: The computed focal loss.
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.positive_weights)
        p_t = torch.exp(-bce_loss)  
        focal_loss = (1 - p_t) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("Invalid reduction method. Use 'none', 'mean', or 'sum'.")

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        smooth = 1.0
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1.0 - ((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth).clamp(min=1e-6))
        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, positive_weights, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.positive_weights = positive_weights
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(self.positive_weights)

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        scaling_factor = ((focal_loss.detach() + 1e-6) / (dice_loss.detach() + 1e-6)).pow(0.5)
        scaling_factor = scaling_factor.clamp(0.1, 10)
        combined_loss = self.dice_weight * (dice_loss * scaling_factor) + (1 - self.dice_weight) * focal_loss
        return combined_loss

def find_best_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    best_thresh = 0.5
    best_f1 = 0
    
    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh, best_f1

def calculate_top_n_edit_accuracy(atom_probs, atom_labels, bond_probs, bond_labels, edge_index, top_n=1):
    candidate_edits = []
    scores = []

    for idx, prob in enumerate(atom_probs):
        candidate_edits.append(f'atom_{idx}')
        scores.append(prob)

    seen_bonds = set()
    for idx, prob in enumerate(bond_probs):
        u, v = edge_index[idx]
        key = f"bond_{min(u, v)}_{max(u, v)}"
        if key not in seen_bonds:
            candidate_edits.append(key)
            scores.append(prob)
            seen_bonds.add(key)
            
    true_edits = set()
    for idx, label in enumerate(atom_labels):
        if label == 1:
            true_edits.add(f'atom_{idx}')
    for idx, label in enumerate(bond_labels):
        if label in [1, 2]:   # 1: bond break, 2: bond order change
            u, v = edge_index[idx]
            true_edits.add( f"bond_{min(u, v)}_{max(u, v)}")  

    scores = np.array(scores)
    sorted_indices = np.argsort(-scores)[:top_n]
    top_n_preds = [candidate_edits[i] for i in sorted_indices]

    hit = any(pred in true_edits for pred in top_n_preds)

    return int(hit)

