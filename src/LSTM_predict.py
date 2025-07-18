import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os

MODEL_PATH = r"\lstm.pth"
DATA_PATH = r"\data.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
PAD_VALUE = -1.0


class LSTMDissDataset(Dataset):
    def __init__(self, X_list, lengths):
        self.X = pad_sequence(X_list, batch_first=True, padding_value=PAD_VALUE)
        self.lengths = lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.lengths[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        out = self.fc(out).squeeze(-1)
        return out


def create_batch_mask(lengths, max_len):
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask


def load_new_data(path):
    df = pd.read_csv(path)
    X_seq, ids, times = [], [], []
    for fid, group in df.groupby("formulation_id"):
        group = group.sort_values("Output_time")
        X = group.drop(columns=["formulation_id", "Output_diss_fraction"], errors='ignore').values
        times.append(group["Output_time"].values)
        X_seq.append(torch.tensor(X, dtype=torch.float32))
        ids.append(fid)
    return X_seq, np.array(ids), times


def predict():
    print("Loading data and model...")
    X_seq, ids, times = load_new_data(DATA_PATH)
    lengths = torch.tensor([len(x) for x in X_seq], dtype=torch.int64)
    dataset = LSTMDissDataset(X_seq, lengths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_seq[0].shape[1]
    model = LSTMModel(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    predictions = []
    with torch.no_grad():
        for xb, lengths in loader:
            xb = xb.to(DEVICE)
            pred = model(xb, lengths).cpu().numpy()
            for p, l in zip(pred, lengths):
                predictions.append(p[:l].tolist())

    # 保存预测结果
    print("Saving predictions...")
    result_rows = []
    for fid, time_arr, pred_arr in zip(ids, times, predictions):
        for t, p in zip(time_arr, pred_arr):
            result_rows.append([fid, t, p])

    result_df = pd.DataFrame(result_rows, columns=["formulation_id", "Output_time", "Predicted_diss_fraction"])
    result_df.to_csv(r"\predicted_dissolution.csv", index=False)
    print("Prediction saved to predicted_dissolution.csv")


if __name__ == "__main__":
    predict()
