import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_COLUMNS, TRAIN_SEASONS, TEST_SEASON

class SequenceDataset(Dataset):
    """
    Creates sequences of past performances for Home and Away teams.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], seq_length: int = 5):
        self.df = df.copy().sort_values("Date").reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_length = seq_length
        
        # Build team histories
        self.team_history = {}
        for team in set(self.df["HomeTeam"]).union(set(self.df["AwayTeam"])):
            # Extract matches involving the team
            team_matches = self.df[(self.df["HomeTeam"] == team) | (self.df["AwayTeam"] == team)].sort_values("Date")
            self.team_history[team] = team_matches[self.feature_cols].values.astype(np.float32)
            
        # We need a quick way to map a specific match index to the team's history up to that point
        # But for simplicity, we will just iterate and build the dataset in memory
        self.X_home = []
        self.X_away = []
        self.y = []
        
        # We need to map team name and date to their history
        team_idx = {team: 0 for team in self.team_history.keys()}
        
        for idx, row in self.df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            
            h_idx = team_idx[home]
            a_idx = team_idx[away]
            
            # Get past sequence for home team
            if h_idx >= self.seq_length:
                h_seq = self.team_history[home][h_idx - self.seq_length : h_idx]
            else:
                # Pad with first available if not enough history
                h_seq = np.zeros((self.seq_length, len(self.feature_cols)), dtype=np.float32)
                if h_idx > 0:
                    h_seq[-h_idx:] = self.team_history[home][:h_idx]
                    h_seq[:-h_idx] = self.team_history[home][0]
                    
            # Get past sequence for away team
            if a_idx >= self.seq_length:
                a_seq = self.team_history[away][a_idx - self.seq_length : a_idx]
            else:
                a_seq = np.zeros((self.seq_length, len(self.feature_cols)), dtype=np.float32)
                if a_idx > 0:
                    a_seq[-a_idx:] = self.team_history[away][:a_idx]
                    a_seq[:-a_idx] = self.team_history[away][0]
                    
            self.X_home.append(h_seq)
            self.X_away.append(a_seq)
            self.y.append(row[TARGET_COLUMNS].values.astype(np.float32))
            
            team_idx[home] += 1
            team_idx[away] += 1
            
        self.X_home = np.array(self.X_home)
        self.X_away = np.array(self.X_away)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_home[idx]),
            torch.tensor(self.X_away[idx]),
            torch.tensor(self.y[idx])
        )

class DualLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, out_dim: int = 6):
        super().__init__()
        self.lstm_home = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_away = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, out_dim)
        
    def forward(self, x_home, x_away):
        _, (h_home, _) = self.lstm_home(x_home)
        _, (h_away, _) = self.lstm_away(x_away)
        
        # h_n shape: (num_layers, batch, hidden_dim). Take the last layer.
        h_home = h_home[-1]
        h_away = h_away[-1]
        
        merged = torch.cat((h_home, h_away), dim=1)
        out = self.relu(self.fc1(merged))
        out = self.fc2(out)
        
        # We predict counts (goals, corners, shots), so ReLU ensures non-negative
        return torch.relu(out)

def train_lstm_pipeline():
    from src.data_ingestion import load_all_data
    from src.feature_engineering import build_features
    from sklearn.metrics import mean_absolute_error
    import sys
    
    print("Loading data...")
    df = load_all_data()
    enriched_df, _, feature_cols = build_features(df)
    
    print("Creating sequential datasets...")
    train_mask = enriched_df["Season"].isin(TRAIN_SEASONS)
    test_mask = enriched_df["Season"] == TEST_SEASON
    
    # Scale features
    scaler = StandardScaler()
    enriched_df[feature_cols] = scaler.fit_transform(enriched_df[feature_cols])
    
    # Re-split after scaling (since SequenceDataset needs sequential access)
    train_df = enriched_df[train_mask]
    test_df = enriched_df[test_mask]
    
    train_dataset = SequenceDataset(train_df, feature_cols, seq_length=5)
    test_dataset = SequenceDataset(test_df, feature_cols, seq_length=5)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("Training Dual LSTM...")
    model = DualLSTM(input_dim=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_h, x_a, y in train_loader:
            optimizer.zero_grad()
            preds = model(x_h, x_a)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
        
    print("Evaluating LSTM...")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_h, x_a, y in test_loader:
            preds = model(x_h, x_a)
            all_preds.append(preds.numpy())
            all_targets.append(y.numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    mae = mean_absolute_error(all_targets, all_preds)
    print(f"\n======================================")
    print(f"LSTM Overall MAE: {mae:.4f}")
    print(f"======================================")
    
    # Compare with current best
    print("To check if it beats the NN, compare this score against the NeuralNet baseline MAE!")

if __name__ == "__main__":
    train_lstm_pipeline()
