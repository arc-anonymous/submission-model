import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from model import ARC
from utils import compute_metrics, calculate_accuracy, get_symmetry_aware_atom_mapping, get_symmetry_aware_accuracy, calculate_top_n_edit_accuracy
from pairdata import *
from dataset import *
from utils_data import *
from plots import plot_atom_bond_roc_pr_curves


def load_model(params, model_path, device):
    model = ARC(
        input_dim=params['node_feature_dim'],
        hidden_dim=params['embedding_dim'],
        edge_attr_dim=params['edge_feature_dim'],
        edge_hidden_dim=params['edge_hidden_dim'],
        dual_edge_attr_dim=params['node_feature_dim'],
        n_heads=params['n_heads'],
        n_layers=params['num_layers'],
        cat=params['cat'],
        learning_skip_connection=True
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pt')))
    model.eval()
    return model


def evaluate_model_on_testset(model, test_loader, threshold_atoms, threshold_bonds, device):
    metrics = {
        "amap_acc": [],
        "sym_amap_acc": [],
        "atom_f1": [], "atom_recall": [], "atom_precision": [], "atom_auc": [], "atom_pr_auc": [],
        "bond_f1": [], "bond_recall": [], "bond_precision": [], "bond_auc": [], "bond_pr_auc": [],
        "top1": [], "top3": [], "top5": []
    }
    all_atom_probs, all_bond_probs = [], []
    all_atom_labels, all_bond_labels = [], []
    all_edge_indices = []

    for data in tqdm(test_loader, desc="Evaluating", ncols=80):
        data = data.to(device)
        dual_data = get_dual_pairdata(data).to(device)

        soft_matching, atom_logits, bond_logits = model(
            data.x_p, data.edge_index_p, data.edge_attr_p, data.batch_p,
            data.x_r, data.edge_index_r, data.edge_attr_r, data.batch_r,
            dual_data.edge_index_p, dual_data.edge_attr_p, 
            mapping_matrix=None, use_ground_truth=False, symmetry_data=data
        )

        # AMAP Accuracy
        amap_acc = calculate_accuracy(soft_matching, data)
        sym_mapping = get_symmetry_aware_atom_mapping(soft_matching, data)
        sym_amap_acc = get_symmetry_aware_accuracy(sym_mapping, data)

        metrics["amap_acc"].append(amap_acc * 100)
        metrics["sym_amap_acc"].append(sym_amap_acc * 100)

        atom_labels = data.rc_atoms.float()
        bond_labels = data.rc_bonds.long()
        bond_binary_labels = (bond_labels > 0).long()

        atom_probs = torch.sigmoid(atom_logits).cpu().detach().numpy()
        bond_probs = torch.sigmoid(bond_logits).cpu().detach().numpy()

        y_true_atom = atom_labels.cpu().numpy()
        y_pred_probs_atom = atom_probs

        y_true_bond = bond_binary_labels.cpu().numpy()
        y_pred_probs_bond = bond_probs

        atom_metrics = compute_metrics(y_true_atom, y_pred_probs_atom, threshold=threshold_atoms, task_name="atom")
        bond_metrics = compute_metrics(y_true_bond, y_pred_probs_bond, threshold=threshold_bonds, task_name="bond")

        for k in atom_metrics:
            metrics[k].append(atom_metrics[k])
        for k in bond_metrics:
            metrics[k].append(bond_metrics[k])

        edge_index = data.edge_index_p.cpu().detach().numpy().T
        hit1 = calculate_top_n_edit_accuracy(atom_probs, atom_labels, bond_probs, bond_labels, edge_index, top_n=1)
        hit3 = calculate_top_n_edit_accuracy(atom_probs, atom_labels, bond_probs, bond_labels, edge_index, top_n=3)
        hit5 = calculate_top_n_edit_accuracy(atom_probs, atom_labels, bond_probs, bond_labels, edge_index, top_n=5)

        metrics["top1"].append(hit1)
        metrics["top3"].append(hit3)
        metrics["top5"].append(hit5)

        all_atom_probs.append(atom_probs)
        all_bond_probs.append(bond_probs)
        all_atom_labels.append(y_true_atom)
        all_bond_labels.append(bond_labels.detach().cpu().numpy())
        all_edge_indices.append(edge_index)

    return metrics, all_atom_probs, all_bond_probs, all_atom_labels, all_bond_labels


def summarize_metrics_to_df(metrics):
    summary = {
        "AMAP Accuracy": [np.mean(metrics["amap_acc"]), np.std(metrics["amap_acc"]) / np.sqrt(len(metrics["amap_acc"]))],
        "Symmetry-Aware AMAP Accuracy": [np.mean(metrics["sym_amap_acc"]), np.std(metrics["sym_amap_acc"]) / np.sqrt(len(metrics["sym_amap_acc"]))],
        "Atom F1": [np.mean(metrics["atom_f1"]*100), np.std(metrics["atom_f1"])],
        "Atom Recall": [np.mean(metrics["atom_recall"]*100), np.std(metrics["atom_recall"])],
        "Atom Precision": [np.mean(metrics["atom_precision"]*100), np.std(metrics["atom_precision"])],
        "Atom AUC": [np.mean(metrics["atom_auc"]*100), np.std(metrics["atom_auc"])],
        "Atom PR AUC": [np.mean(metrics["atom_pr_auc"]*100), np.std(metrics["atom_pr_auc"])],
        "Bond F1": [np.mean(metrics["bond_f1"]*100), np.std(metrics["bond_f1"])],
        "Bond Recall": [np.mean(metrics["bond_recall"]*100), np.std(metrics["bond_recall"])],
        "Bond Precision": [np.mean(metrics["bond_precision"]*100), np.std(metrics["bond_precision"])],
        "Bond AUC": [np.mean(metrics["bond_auc"]*100), np.std(metrics["bond_auc"])],
        "Bond PR AUC": [np.mean(metrics["bond_pr_auc"]*100), np.std(metrics["bond_pr_auc"])],
        "Top-1 Edit Accuracy": [np.mean(metrics["top1"]*100), np.std(metrics["top1"])],
        "Top-3 Edit Accuracy": [np.mean(metrics["top3"]*100), np.std(metrics["top3"])],
        "Top-5 Edit Accuracy": [np.mean(metrics["top5"]*100), np.std(metrics["top5"])]
    }
    df = pd.DataFrame(summary, index=["Mean", "Std"]).T.round(3)
    return df


def main():
    threshold_atoms = 0.566
    threshold_bonds = 0.550
    dataset_path = '../datasets/processed/'
    model_path = 'results/arc'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data and model...")
    test_set = torch.load(os.path.join(dataset_path, 'test.pt'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    with open(f'{model_path}/hyperparameters.json', 'r') as f:
        params = json.load(f)

    model = load_model(params, model_path, device)
    metrics, all_atom_probs, all_bond_probs, all_atom_labels, all_bond_labels = evaluate_model_on_testset(
                                                                                model, test_loader, 
                                                                                threshold_atoms, threshold_bonds, device)
    df_metrics = summarize_metrics_to_df(metrics)
    print(df_metrics)
    path = 'results'
    os.makedirs(path, exist_ok=True)
    df_metrics.to_csv(f"{path}/eval_summary.csv")


    os.makedirs("plots", exist_ok=True)
    plot_atom_bond_roc_pr_curves(
        np.concatenate(all_atom_labels),
        np.concatenate(all_atom_probs),
        (np.concatenate(all_bond_labels) > 0).astype(int),
        np.concatenate(all_bond_probs)
    )
    print("Evaluation complete. Summary saved to 'results/eval_summary.csv'.")
if __name__ == "__main__":
    main()
