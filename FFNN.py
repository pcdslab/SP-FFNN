import os
import sys
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support


TRAIN_CSV = '/lclhome/zsaeed/MAPS/data/cell_phenotyping/train.csv'
VALID_CSV = '/lclhome/zsaeed/MAPS/data/cell_phenotyping/valid.csv'
RESULTS_DIR = './results_ffnn'
LABEL_COL = 'cell_label'
NUM_FEATURES = 50
NUM_CLASSES = 16
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DROPOUT = 0.2
HIDDEN_DIM = 512
N_BLOCKS = 4
MAX_EPOCHS = 500
MIN_EPOCHS = 250
PATIENCE = 100
SEED = 7325111
NUM_WORKERS = 4
VERBOSE = 1


class CellExpressionCSV(Dataset):
    def __init__(self, csv_path, is_train=False, mean=None, std=None, label_col='cell_label'):
        self.csv_path = csv_path
        self.label_col = label_col
        self.df = pd.read_csv(csv_path)
        if label_col not in self.df.columns:
            raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

        self.feature_cols = [c for c in self.df.columns if c != label_col]
        x = self.df[self.feature_cols].astype(np.float64).values
        y = self.df[label_col].astype(int).values

        if is_train:
            self.mean = x.mean(axis=0)
            self.std = x.std(axis=0)
            self.std[self.std == 0] = 1.0
        else:
            if mean is None or std is None:
                self.mean = np.zeros(x.shape[1], dtype=np.float64)
                self.std = np.ones(x.shape[1], dtype=np.float64)
            else:
                self.mean = np.asarray(mean, dtype=np.float64)
                self.std = np.asarray(std, dtype=np.float64)
                self.std[self.std == 0] = 1.0

        self.x = ((x - self.mean) / self.std).astype(np.float64)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)

    @staticmethod
    def get_data_loader(dataset, batch_size=128, is_train=False, num_workers=0):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = torch.relu(out)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1, n_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        self.fc_mid = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_mid = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout_mid = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        if x.dtype != torch.float64:
            x = x.double()
        out = self.fc_in(x)
        out = self.bn_in(out)
        out = torch.relu(out)
        for block in self.blocks:
            out = block(out)
        out = self.fc_mid(out)
        out = self.bn_mid(out)
        out = torch.relu(out)
        out = self.dropout_mid(out)
        logits = self.fc_out(out)
        probs = torch.softmax(logits, dim=1)
        return logits, probs


class Trainer:
    def __init__(self):
        self.model_checkpoint_path = None
        self.results_dir = RESULTS_DIR
        self.num_features = NUM_FEATURES
        self.num_classes = NUM_CLASSES
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.dropout = DROPOUT
        self.max_epochs = MAX_EPOCHS
        self.min_epochs = MIN_EPOCHS
        self.patience = PATIENCE
        self.num_workers = NUM_WORKERS
        self.seed = SEED
        self.verbose = VERBOSE
        self.hidden_dim = HIDDEN_DIM
        self.n_blocks = N_BLOCKS
        self.label_col = LABEL_COL

        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.counter = 0
        self.lowest_loss = np.inf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_seed(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def init_data_loader(self, data_path, is_train=False, mean=None, std=None):
        dataset = CellExpressionCSV(data_path, is_train=is_train, mean=mean, std=std, label_col=self.label_col)
        if dataset.x.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, found {dataset.x.shape[1]} in {data_path}")
        return CellExpressionCSV.get_data_loader(dataset, batch_size=self.batch_size, is_train=is_train,
                                                 num_workers=self.num_workers)

    def init_model(self):
        self.model = MLP(
            input_dim=self.num_features,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=self.dropout,
            n_blocks=self.n_blocks,
        )
        self.model.to(self.device, dtype=torch.float64)

    def init_optimizer(self):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)

    def init_loss_function(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def save_model(self, mean, std):
        os.makedirs(self.results_dir, exist_ok=True)
        save_dict = {
            'model_parameters': self.model.state_dict(),
            'train_data_mean': mean,
            'train_data_std': std,
        }
        ckpt = os.path.join(self.results_dir, 'best_checkpoint.pt')
        torch.save(save_dict, ckpt)
        self.model_checkpoint_path = ckpt

    def _compute_metrics(self, gt_labels, pred_labels, pred_probs):
        acc = accuracy_score(gt_labels, pred_labels)
        try:
            auc = roc_auc_score(gt_labels, pred_probs, multi_class='ovo', labels=[i for i in range(self.num_classes)])
        except Exception:
            auc = np.nan
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            gt_labels, pred_labels, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            gt_labels, pred_labels, average='weighted', zero_division=0
        )
        return {
            'acc': acc,
            'auc': auc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
        }

    def fit(self, train_dataset_path, valid_dataset_path):
        self.counter = 0
        self.lowest_loss = np.inf
        self.set_seed()
        self.init_model()
        self.init_optimizer()
        self.init_loss_function()

        train_dl = self.init_data_loader(train_dataset_path, is_train=True)
        train_mean = train_dl.dataset.mean
        train_std = train_dl.dataset.std
        valid_dl = self.init_data_loader(valid_dataset_path, mean=train_mean, std=train_std)

        os.makedirs(self.results_dir, exist_ok=True)
        result_dict = {
            'train_loss': [], 'valid_loss': [],
            'train_acc': [], 'valid_acc': [],
            'train_auc': [], 'valid_auc': [],
            'train_precision_macro': [], 'valid_precision_macro': [],
            'train_recall_macro': [], 'valid_recall_macro': [],
            'train_f1_macro': [], 'valid_f1_macro': [],
            'train_precision_weighted': [], 'valid_precision_weighted': [],
            'train_recall_weighted': [], 'valid_recall_weighted': [],
            'train_f1_weighted': [], 'valid_f1_weighted': [],
        }

        for epoch in range(self.max_epochs):
            start_time = time.time()
            train_loss, train_metrics = self.train_loop(train_dl)
            valid_loss, valid_metrics = self.valid_loop(valid_dl)

            print(
                f"Epoch {epoch} | "
                f"train_loss={train_loss:.4f}, train_acc={train_metrics['acc']:.4f}, train_auc={train_metrics['auc']:.4f}, train_f1_macro={train_metrics['f1_macro']:.4f} | "
                f"valid_loss={valid_loss:.4f}, valid_acc={valid_metrics['acc']:.4f}, valid_auc={valid_metrics['auc']:.4f}, valid_f1_macro={valid_metrics['f1_macro']:.4f}"
            )

            result_dict['train_loss'].append(train_loss)
            result_dict['valid_loss'].append(valid_loss)
            for k in ['acc', 'auc', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
                result_dict[f'train_{k}'].append(train_metrics[k])
                result_dict[f'valid_{k}'].append(valid_metrics[k])

            if self.lowest_loss > valid_loss:
                print('-------------------- Saving best model --------------------')
                self.save_model(train_mean, train_std)
                self.lowest_loss = valid_loss
                self.counter = 0
            else:
                self.counter += 1
                print(f'Validation loss not decreased for {self.counter} epoch(s)')

            pd.DataFrame.from_dict(result_dict).to_csv(os.path.join(self.results_dir, 'training_logs.csv'), index=False)

            if self.verbose == 1:
                conf = pd.DataFrame(
                    confusion_matrix(valid_metrics['gt_labels'], valid_metrics['pred_labels'], labels=[i for i in range(self.num_classes)]),
                    index=[f'class_{i}' for i in range(self.num_classes)],
                    columns=[f'class_{i}' for i in range(self.num_classes)]
                )
                conf.to_csv(os.path.join(self.results_dir, f'valid_confusion_matrix_epoch_{epoch}.csv'))

            if (self.counter > self.patience) and (epoch >= self.min_epochs):
                print('Early stopping triggered.')
                break

            total_time = time.time() - start_time
            print(f'Time to process epoch({epoch}): {total_time/60:.4f} minutes\n')

    def train_loop(self, data_loader):
        total_loss = 0.0
        gt_labels, pred_labels = [], []
        pred_probs = []
        self.model.train()
        batch_count = len(data_loader)

        for batch_idx, (features_batch, label_batch) in enumerate(data_loader):
            gt_labels.extend(label_batch.cpu().numpy().tolist())
            features_batch = features_batch.to(self.device)
            label_batch = label_batch.to(self.device)

            logits, probs = self.model(features_batch)
            self.optimizer.zero_grad()
            loss = self.loss_fn(logits, label_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            p_labels = torch.argmax(probs, dim=1).detach().cpu().numpy().tolist()
            p_probs = probs.detach().cpu().numpy()
            pred_labels.extend(p_labels)
            pred_probs.append(p_probs)

            sys.stdout.write(f'\rTraining Batch {batch_idx + 1}/{batch_count}, avg loss: {total_loss / (batch_idx + 1):.4f}')
            sys.stdout.flush()
        print()

        pred_probs = np.concatenate(pred_probs, axis=0)
        metrics = self._compute_metrics(gt_labels, pred_labels, pred_probs)
        metrics['gt_labels'] = gt_labels
        metrics['pred_labels'] = pred_labels
        metrics['pred_probs'] = pred_probs
        return total_loss / batch_count, metrics

    def valid_loop(self, data_loader):
        total_loss = 0.0
        gt_labels, pred_labels = [], []
        pred_probs = []
        self.model.eval()
        batch_count = len(data_loader)
        with torch.no_grad():
            for batch_idx, (features_batch, label_batch) in enumerate(data_loader):
                gt_labels.extend(label_batch.cpu().numpy().tolist())
                features_batch = features_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                logits, probs = self.model(features_batch)
                loss = self.loss_fn(logits, label_batch)
                total_loss += loss.item()
                p_labels = torch.argmax(probs, dim=1).detach().cpu().numpy().tolist()
                p_probs = probs.detach().cpu().numpy()
                pred_labels.extend(p_labels)
                pred_probs.append(p_probs)
                sys.stdout.write(f'\rValidation Batch {batch_idx + 1}/{batch_count}, avg loss: {total_loss / (batch_idx + 1):.4f}')
                sys.stdout.flush()
        print()

        pred_probs = np.concatenate(pred_probs, axis=0)
        metrics = self._compute_metrics(gt_labels, pred_labels, pred_probs)
        metrics['gt_labels'] = gt_labels
        metrics['pred_labels'] = pred_labels
        metrics['pred_probs'] = pred_probs
        return total_loss / batch_count, metrics


if __name__ == '__main__':
    print('Starting training with embedded paths...')
    print(f'TRAIN_CSV: {TRAIN_CSV}')
    print(f'VALID_CSV: {VALID_CSV}')
    print(f'RESULTS_DIR: {RESULTS_DIR}')
    trainer = Trainer()
    trainer.fit(TRAIN_CSV, VALID_CSV)
