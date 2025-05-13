import torch
from torch.optim import AdamW
import numpy as np
from torch_geometric.loader import DataLoader
import os
import argparse
from model import *
from utils import *
from pairdata import *
from dataset import *
from utils_data import *
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='arc', help='Name of the model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--node_feature_dim', type=int, default=71)
    parser.add_argument('--edge_feature_dim', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--edge_hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--cat', action='store_true', help='Whether to concatenate features in the model')
    parser.add_argument('--lambda_amap', type=float, default=0.25, help='weight for atom mapping loss')
    parser.add_argument('--lambda_atom', type=float, default=0.25, help='weight for atom loss')
    parser.add_argument('--lambda_bond', type=float, default=0.25, help='weight for bond loss')
    parser.add_argument('--dice_weight', type=float, default=0.4, help='weight for tunning combined loss')
    parser.add_argument('--pos_weight_atom', type=float, default=7.23, help='positive weight for atom loss')
    parser.add_argument('--pos_weight_bond', type=float, default=7.5, help='positive weight for bond binary loss')
    parser.add_argument('--threshold_atoms', type=float, default= 0.566, help='threshold for atom prediction')
    parser.add_argument('--threshold_bonds', type=float, default=0.550, help='threshold for binary bond prediction')
    parser.add_argument('--T_0', type=int, default=10, help='Number of epochs before the first restart')
    parser.add_argument('--T_mult', type=int, default=2, help='Number of epochs between restarts')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate')
    return parser.parse_args()

def setup_environment():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')    
    os.makedirs('results/arc', exist_ok=True)
    return args, device

def get_dataloaders(args):
    dataset_path = '../datasets/processed/'
    train_set = torch.load(os.path.join(dataset_path, 'train.pt'))
    validation_set = torch.load(os.path.join(dataset_path, 'valid.pt'))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=['x_r', 'x_p'])
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, follow_batch=['x_r', 'x_p'])

    return train_loader, val_loader

def build_model(args, device):
    encoder = ARC(
        input_dim=args.node_feature_dim,
        hidden_dim=args.embedding_dim,
        edge_attr_dim=args.edge_feature_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        dual_edge_attr_dim = args.node_feature_dim,
        n_heads=args.n_heads,
        n_layers=args.num_layers,
        cat=args.cat,
        learning_skip_connection=True,
    ).to(device)

    pos_weight_atom = torch.tensor([args.pos_weight_atom], device=device)
    pos_weight_bond = torch.tensor([args.pos_weight_bond], device=device)

    atom_loss_fn = CombinedLoss(positive_weights=pos_weight_atom, dice_weight=args.dice_weight) 
    bond_loss_fn = CombinedLoss(positive_weights=pos_weight_bond, dice_weight=args.dice_weight)

    optimizer = AdamW(encoder.parameters() , lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
    return encoder, atom_loss_fn, bond_loss_fn, optimizer, scheduler

def initialize_epoch_result():
    return {
        'loss_all': 0.0, 'loss_amap': 0.0, 'loss_atom': 0.0, 'loss_bond': 0.0,
        'amap_acc': 0.0, 'count': 0,
        'atom_probs':[], 'atom_preds': [], 'atom_labels': [],
        'bond_probs':[],'bond_preds': [], 'bond_labels': []}

def update_epoch_result(epoch_result,loss, loss_amap, loss_atom, loss_bond, amap_acc, 
                        y_true_atom, y_pred_probs_atom, y_preds_atom, y_true_bond, y_pred_probs_bond, y_preds_bond):

    epoch_result['loss_all'] += loss.item()
    epoch_result['loss_amap'] += loss_amap.item()
    epoch_result['loss_atom'] += loss_atom.item()
    epoch_result['loss_bond'] += loss_bond.item()
    epoch_result['amap_acc'] += amap_acc
    epoch_result['atom_probs'].extend(np.ravel(y_pred_probs_atom).tolist())
    epoch_result['atom_preds'].extend(np.ravel(y_preds_atom).tolist())
    epoch_result['atom_labels'].extend(np.ravel(y_true_atom).tolist())

    epoch_result['bond_probs'].extend(np.ravel(y_pred_probs_bond).tolist())
    epoch_result['bond_preds'].extend(np.ravel(y_preds_bond).tolist())
    epoch_result['bond_labels'].extend(np.ravel(y_true_bond).tolist())

    epoch_result['count'] += 1

    
def compute_epoch_metrics(args,epoch_result, mode="train"):
    atom_probs = epoch_result['atom_probs']
    atom_preds = epoch_result['atom_preds']
    atom_labels = epoch_result['atom_labels']
    bond_probs =  epoch_result['bond_probs']
    bond_labels = epoch_result['bond_labels']
    bond_preds = epoch_result['bond_preds']

    atom_metrics = compute_metrics(np.array(atom_labels), np.array(atom_probs), threshold=args.threshold_atoms, task_name='atom')
    bond_metrics = compute_metrics(np.array(bond_labels),np.array(bond_probs), threshold=args.threshold_bonds, task_name='bond')

    return {
        f'{mode}_all_loss': epoch_result['loss_all'] / epoch_result['count'],
        f'{mode}_amap_loss': epoch_result['loss_amap'] / epoch_result['count'],
        f'{mode}_atom_loss': epoch_result['loss_atom'] / epoch_result['count'],
        f'{mode}_bond_loss': epoch_result['loss_bond'] / epoch_result['count'],
        f'{mode}_amap_acc': epoch_result['amap_acc'] / epoch_result['count'],
        f'{mode}_atom_acc': np.mean(np.array(atom_preds) == np.array(atom_labels)),
        f'{mode}_bond_acc': np.mean(np.array(bond_preds) == np.array(bond_labels)),
        f'{mode}_atom_f1': atom_metrics['atom_f1'],
        f'{mode}_atom_auc': atom_metrics['atom_auc'],
        f'{mode}_atom_precision': atom_metrics['atom_precision'],
        f'{mode}_atom_recall': atom_metrics['atom_recall'],
        f'{mode}_bond_f1': bond_metrics['bond_f1'],
        f'{mode}_bond_auc': bond_metrics['bond_auc'],
        f'{mode}_bond_precision': bond_metrics['bond_precision'],
        f'{mode}_bond_recall': bond_metrics['bond_recall'],
    }
def train_on_epoch(args, encoder, optimizer, train_loader, atom_loss_fn, bond_loss_fn, device):
    encoder.train()
    epoch_result = initialize_epoch_result()
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        dual_data = get_dual_pairdata(data)  
        dual_data = dual_data.to(device)  

        soft_matching, atom_change_logits, bond_logits = encoder(
            data.x_p, data.edge_index_p, data.edge_attr_p, data.batch_p,
            data.x_r, data.edge_index_r, data.edge_attr_r, data.batch_r,
            dual_data.edge_index_p, dual_data.edge_attr_p, mapping_matrix=data.p2r_mapper)

        atom_labels = data.rc_atoms.float()
        bond_labels = data.rc_bonds.long()
        bond_binary_labels = (bond_labels > 0).long()
       
        loss_amap = compute_loss(soft_matching, data)  
        loss_atom = atom_loss_fn(atom_change_logits, atom_labels)
        loss_bond =  bond_loss_fn(bond_logits, bond_binary_labels.float())
        
        loss = (args.lambda_amap * loss_amap +
        args.lambda_atom * loss_atom +
        args.lambda_bond * loss_bond )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_norm)
        optimizer.step()
        amap_acc = calculate_accuracy(soft_matching, data)

        y_true_atom = atom_labels.cpu().detach().numpy()
        y_pred_probs_atom = torch.sigmoid(atom_change_logits).cpu().detach().numpy()
        y_preds_atom = (y_pred_probs_atom > args.threshold_atoms).astype(int)

        y_true_bond = bond_binary_labels.cpu().detach().numpy()
        y_pred_probs_bond = torch.sigmoid(bond_logits).cpu().detach().numpy()
        y_preds_bond = (y_pred_probs_bond > args.threshold_bonds).astype(int)

        update_epoch_result(epoch_result,loss, loss_amap, loss_atom, loss_bond, amap_acc, 
                        y_true_atom, y_pred_probs_atom, y_preds_atom, y_true_bond, y_pred_probs_bond, y_preds_bond)
        
    return compute_epoch_metrics(args, epoch_result, mode="train")

def valid_one_epoch(args, encoder, validation_loader, atom_loss_fn, bond_loss_fn, device):
    epoch_result = initialize_epoch_result()
    encoder.eval()

    with torch.no_grad():

        for data in validation_loader:
            data = data.to(device)
            dual_data = get_dual_pairdata(data) 
            dual_data = dual_data.to(device) 
            soft_matching, atom_change_logits, bond_logits = encoder(
                                    data.x_p, data.edge_index_p, data.edge_attr_p, data.batch_p,
                                    data.x_r, data.edge_index_r, data.edge_attr_r, data.batch_r,
                                    dual_data.edge_index_p, dual_data.edge_attr_p, mapping_matrix=data.p2r_mapper)         

            atom_labels = data.rc_atoms.float()
            bond_labels = data.rc_bonds.long()
            bond_binary_labels = (bond_labels > 0).long()

            loss_amap = compute_loss(soft_matching, data)  
            loss_atom = atom_loss_fn(atom_change_logits, atom_labels)
            loss_bond = bond_loss_fn(bond_logits, bond_binary_labels.float())
            
            loss = (args.lambda_amap * loss_amap +
            args.lambda_atom * loss_atom +
            args.lambda_bond * loss_bond)

            amap_acc = calculate_accuracy(soft_matching, data)

            y_true_atom = atom_labels.cpu().detach().numpy()
            y_pred_probs_atom = torch.sigmoid(atom_change_logits).cpu().detach().numpy()
            y_preds_atom = (y_pred_probs_atom > args.threshold_atoms).astype(int)
            
            y_true_bond = bond_binary_labels.cpu().detach().numpy()
            y_pred_probs_bond = torch.sigmoid(bond_logits).cpu().detach().numpy()
            y_preds_bond = (y_pred_probs_bond > args.threshold_bonds).astype(int)
            update_epoch_result(epoch_result,loss, loss_amap, loss_atom, loss_bond, amap_acc, 
                        y_true_atom, y_pred_probs_atom, y_preds_atom, y_true_bond, y_pred_probs_bond, y_preds_bond)
            
    best_atom_thresh, best_atom_f1 = find_best_threshold(np.array(epoch_result['atom_labels']), np.array(epoch_result['atom_probs']))
    best_bond_thresh, best_bond_f1 = find_best_threshold(np.array(epoch_result['bond_labels']), np.array(epoch_result['bond_probs']))
    print(f"[Val] Best atom threshold: {best_atom_thresh:.3f} | F1: {best_atom_f1:.3f}")
    print(f"[Val] Best bond threshold: {best_bond_thresh:.3f} | F1: {best_bond_f1:.3f}")

    np.save(os.path.join('results', args.model_name, 'best_atom_threshold.npy'), best_atom_thresh)
    np.save(os.path.join('results', args.model_name, 'best_bond_threshold.npy'), best_bond_thresh)

    return compute_epoch_metrics(args,epoch_result, mode="valid")

def train_model(args, encoder, train_loader, validation_loader, atom_loss_fn, bond_loss_fn, optimizer, scheduler, device):
    path = f'results/{args.model_name}'
    os.makedirs(path, exist_ok=True)

    best_valid_loss = float('inf')
    patience_counter = 0
    start_time = datetime.now()

    history = []
    for epoch in range(args.n_epochs):
        print(f'Epoch {epoch + 1}/{args.n_epochs}')

        train_metrics = train_on_epoch(args, encoder, optimizer, train_loader, atom_loss_fn, bond_loss_fn, device)
        valid_metrics = valid_one_epoch(args, encoder, validation_loader, atom_loss_fn, bond_loss_fn, device)
        scheduler.step()
        metrics = {**train_metrics, **valid_metrics}
        history.append(metrics)

        print(f"Train Loss: {metrics['train_all_loss']:.4f} | Val Loss: {metrics['valid_all_loss']:.4f}")
        print(f"Train AMAP: {metrics['train_amap_acc']:.3f} | Val AMAP: {metrics['valid_amap_acc']:.3f}")
        print(f"Train Atom F1: {metrics['train_atom_f1']:.3f} | Val Atom F1: {metrics['valid_atom_f1']:.3f}")
        print(f"Train Bond F1: {metrics['train_bond_f1']:.3f} | Val Bond F1: {metrics['valid_bond_f1']:.3f}")
    
        if metrics['valid_all_loss'] < best_valid_loss:
            best_valid_loss = metrics['valid_all_loss']
            patience_counter = 0
            torch.save(encoder.state_dict(), os.path.join(path, 'best_model.pt'))
            print("Model improved. Saved best model.")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(history, f, indent=4)
    print(f"\nTraining complete. Metrics saved to {os.path.join(path, 'metrics.json')}")

    print(f"Total training time: {str(datetime.now() - start_time)}")

def main():
    args, device = setup_environment()
    encoder, atom_loss_fn,bond_loss_fn, optimizer, scheduler = build_model(args, device)
    train_loader, validation_loader = get_dataloaders(args)
    train_model(args, encoder, train_loader, validation_loader, atom_loss_fn, bond_loss_fn, optimizer, scheduler, device)

if __name__ == "__main__":
    main()  