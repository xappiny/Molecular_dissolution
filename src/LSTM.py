import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# =================== Parameters ======================
EPOCHS = 200
BATCH_SIZE = 16
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
PATIENCE = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PAD_VALUE = -1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"\lstm_model.pth"
DATA_PATH = r"\scaler_features_model_train.csv"

# =================== Load and group data =============
df = pd.read_csv(DATA_PATH)
groups = df.groupby("formulation_id")

X_seq, y_seq, ids = [], [], []
for fid, group in groups:
    group = group.sort_values("Output_time")
    X = group.drop(columns=["formulation_id", "Output_diss_fraction"]).values
    y = group["Output_diss_fraction"].values
    X_seq.append(torch.tensor(X, dtype=torch.float32))
    y_seq.append(torch.tensor(y, dtype=torch.float32))
    ids.append(fid)

X_seq = np.array(X_seq, dtype=object)
y_seq = np.array(y_seq, dtype=object)
ids = np.array(ids)


# =================== Dataset class ====================
class LSTMDissDataset(Dataset):
    def __init__(self, X_list, y_list, pad_value=PAD_VALUE):
        self.lengths = torch.tensor([len(x) for x in X_list], dtype=torch.int64)
        self.X = pad_sequence(X_list, batch_first=True, padding_value=pad_value)
        self.y = pad_sequence(y_list, batch_first=True, padding_value=pad_value)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


# =================== LSTM Model =======================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)

        # FIX: Force the output padding to match the input's sequence length (x.size(1))
        # This ensures the prediction tensor and target tensor always have the same shape.
        out, out_lengths = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        out = self.fc(out).squeeze(-1)
        return out, out_lengths


# =================== Create mask for batch ====================
def create_batch_mask(lengths, max_len):
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask


# =================== Evaluation Function ========================
# REFACTOR: Create a reusable function for evaluation to reduce code duplication.
def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for xb, yb, lengths in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred, _ = model(xb, lengths)

            mask = create_batch_mask(lengths, xb.size(1)).to(device)
            loss = loss_fn(pred[mask], yb[mask])
            total_loss += loss.item()

            # Collect the actual (un-padded) values for metric calculation
            y_true_all.extend(yb[mask].cpu().numpy())
            y_pred_all.extend(pred[mask].cpu().numpy())

    # Calculate metrics on all collected values
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2 = r2_score(y_true_all, y_pred_all)

    return total_loss / len(data_loader), mae, rmse, r2


# =================== Nested CV ========================
outer_mae, outer_rmse, outer_r2 = [], [], []
all_inner_mae, all_inner_rmse, all_inner_r2 = [], [], []
outer_cv = GroupKFold(n_splits=5)
best_val_mae = float('inf')
best_model_state = None
best_input_dim = None

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_seq, y_seq, groups=ids)):
    print(f"\n===== Outer Fold {outer_fold + 1} =====")
    X_train_outer, y_train_outer = X_seq[train_idx], y_seq[train_idx]
    ids_train_outer = ids[train_idx]

    inner_mae_list, inner_rmse_list, inner_r2_list = [], [], []
    inner_cv = GroupKFold(n_splits=5)

    for inner_fold, (train_inner_idx, val_inner_idx) in enumerate(
            inner_cv.split(X_train_outer, y_train_outer, groups=ids_train_outer)):
        print(f"-- Inner Fold {inner_fold + 1}")
        X_train, y_train = X_train_outer[train_inner_idx], y_train_outer[train_inner_idx]
        X_val, y_val = X_train_outer[val_inner_idx], y_train_outer[val_inner_idx]

        train_dataset = LSTMDissDataset(X_train, y_train)
        val_dataset = LSTMDissDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        input_dim = X_train[0].shape[1]
        model = LSTMModel(input_dim=input_dim, hidden_dim=HIDDEN_SIZE).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        loss_fn = nn.MSELoss()

        best_inner_mae = float('inf')
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            model.train()
            for xb, yb, lengths in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred, _ = model(xb, lengths)
                mask = create_batch_mask(lengths, xb.size(1)).to(DEVICE)
                loss = loss_fn(pred[mask], yb[mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # REFACTOR: Use the evaluation function
            _, val_mae, _, _ = evaluate_model(model, val_loader, loss_fn, DEVICE)
            scheduler.step(val_mae)

            if val_mae < best_inner_mae:
                best_inner_mae = val_mae
                epochs_no_improve = 0
                # This logic saves the best model from any inner fold
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_model_state = model.state_dict()
                    best_input_dim = input_dim
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # REFACTOR: Get final metrics for this inner fold from the evaluation function
        _, mae, rmse, r2 = evaluate_model(model, val_loader, loss_fn, DEVICE)
        inner_mae_list.append(mae)
        inner_rmse_list.append(rmse)
        inner_r2_list.append(r2)

    # Aggregate and report inner loop results for this outer fold
    all_inner_mae.extend(inner_mae_list)
    all_inner_rmse.extend(inner_rmse_list)
    all_inner_r2.extend(inner_r2_list)
    print(
        f"Inner CV - MAE: {np.mean(inner_mae_list):.3f}±{np.std(inner_mae_list):.3f}, R2: {np.mean(inner_r2_list):.3f}±{np.std(inner_r2_list):.3f}")

    # Train a new model on the full outer training set
    print("-- Training on full outer fold data --")
    full_train_dataset = LSTMDissDataset(X_train_outer, y_train_outer)
    full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = LSTMDissDataset(X_seq[test_idx], y_seq[test_idx])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    outer_model = LSTMModel(input_dim=X_train_outer[0].shape[1], hidden_dim=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(outer_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):  # A fixed number of epochs is used here, consider early stopping too
        outer_model.train()
        for xb, yb, lengths in full_train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, _ = outer_model(xb, lengths)
            mask = create_batch_mask(lengths, xb.size(1)).to(DEVICE)
            loss = loss_fn(pred[mask], yb[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the outer model on the test set
    _, mae, rmse, r2 = evaluate_model(outer_model, test_loader, loss_fn, DEVICE)
    outer_mae.append(mae)
    outer_rmse.append(rmse)
    outer_r2.append(r2)
    print(f"Outer Fold {outer_fold + 1} Test Set -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

# Save the best model found across all inner validation folds
if best_model_state is not None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(best_model_state, MODEL_PATH)
    print(f"\nBest model from inner loops saved to {MODEL_PATH} with validation MAE: {best_val_mae:.3f}")

# =================== Final report =====================
print("\n===== Final Nested CV Results =====")
print("\nInner Loop Results:")
print(f"MAE:  {np.mean(all_inner_mae):.3f}±{np.std(all_inner_mae):.3f}")
print(f"RMSE: {np.mean(all_inner_rmse):.3f}±{np.std(all_inner_rmse):.3f}")
print(f"R²:   {np.mean(all_inner_r2):.3f}±{np.std(all_inner_r2):.3f}")
print("\nOuter Loop Results:")
print(f"MAE:  {np.mean(outer_mae):.3f}±{np.std(outer_mae):.3f}")
print(f"RMSE: {np.mean(outer_rmse):.3f}±{np.std(outer_rmse):.3f}")
print(f"R²:   {np.mean(outer_r2):.3f}±{np.std(outer_r2):.3f}")
