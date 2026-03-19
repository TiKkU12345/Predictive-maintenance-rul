import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
from preprocess import load_data, get_features

BASE_PATH = r'D:\D drive\Ml Sys\archive\CMaps'
MODEL_DIR = r'D:\D drive\Ml Sys\models'
SEQUENCE_LEN = 30
os.makedirs(MODEL_DIR, exist_ok=True)

def create_sequences(df, drop_cols, seq_len):
    feature_cols = [c for c in df.columns if c not in drop_cols + ['RUL']]
    X_seqs, y_seqs = [], []
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit].reset_index(drop=True)
        features = unit_df[feature_cols].values
        rul      = unit_df['RUL'].values
        for i in range(len(unit_df) - seq_len):
            X_seqs.append(features[i:i+seq_len])
            y_seqs.append(rul[i+seq_len])
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

def prepare_test_sequences(test, drop_cols, scaler, seq_len):
    feature_cols = [c for c in test.columns if c not in drop_cols + ['RUL', 'unit', 'cycle']]
    X_seqs = []
    for unit in test['unit'].unique():
        unit_df = test[test['unit'] == unit].reset_index(drop=True)
        features = scaler.transform(unit_df[feature_cols].values).astype(np.float32)
        if len(features) >= seq_len:
            X_seqs.append(features[-seq_len:])
        else:
            pad = np.zeros((seq_len - len(features), features.shape[1]), dtype=np.float32)
            X_seqs.append(np.vstack([pad, features]))
    return np.array(X_seqs, dtype=np.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.fc    = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x, _ = self.lstm3(x)
        return self.fc(x[:, -1, :]).squeeze()

# Load & prep
train, test, test_last = load_data(BASE_PATH)
drop_cols = get_features()
train['RUL'] = train['RUL'].clip(upper=125)

feature_cols = [c for c in train.columns if c not in drop_cols + ['RUL']]
scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])

X_train, y_train = create_sequences(train, drop_cols, SEQUENCE_LEN)
print(f'Training sequences: {X_train.shape}')

# DataLoader
dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
loader  = DataLoader(dataset, batch_size=128, shuffle=True)

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = LSTMModel(X_train.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
criterion = nn.MSELoss()

best_loss = float('inf')
patience_counter = 0
PATIENCE = 15

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    scheduler.step(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), f'{MODEL_DIR}/lstm_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break

print(f'\nBest training loss: {best_loss:.4f}')

# Evaluate
model.load_state_dict(torch.load(f'{MODEL_DIR}/lstm_model.pt'))
model.eval()
X_test = prepare_test_sequences(test, drop_cols, scaler, SEQUENCE_LEN)
with torch.no_grad():
    preds = model(torch.tensor(X_test).to(device)).cpu().numpy()

rmse = np.sqrt(mean_squared_error(test_last['RUL'].values, preds))
print(f'LSTM RMSE: {rmse:.2f} cycles')

pickle.dump(scaler, open(f'{MODEL_DIR}/lstm_scaler.pkl', 'wb'))
print('LSTM model and scaler saved.')