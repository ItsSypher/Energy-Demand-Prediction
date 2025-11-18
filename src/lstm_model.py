import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FortumDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(df_cons, seq_length=168, pred_horizon=48):
    """
    Prepares data for LSTM:
    - Normalize consumption
    - Create sequences (sliding window)
    """
    print("Preparing data for LSTM...")
    
    # Pivot to have groups as columns for easier processing if we model all together, 
    # but LSTM usually models one series or uses group ID as feature.
    # Given 112 groups, training one global model with group embedding is best, 
    # or just treating them as independent samples if patterns are similar.
    # Let's try a global model approach where we feed sequences from all groups.
    
    # Melt to long format
    df_long = df_cons.melt(id_vars=['measured_at'], var_name='group_id', value_name='consumption')
    df_long['measured_at'] = pd.to_datetime(df_long['measured_at'], utc=True)
    df_long = df_long.sort_values(['group_id', 'measured_at'])
    
    # Normalize consumption
    scaler = MinMaxScaler()
    df_long['consumption_scaled'] = scaler.fit_transform(df_long[['consumption']])
    
    # Create sequences
    # Input: [samples, seq_length, features]
    # Output: [samples, pred_horizon] (Multi-step forecast)
    
    X_all = []
    y_all = []
    
    groups = df_long['group_id'].unique()
    
    for gid in groups:
        group_data = df_long[df_long['group_id'] == gid]['consumption_scaled'].values
        
        # Sliding window
        # We need at least seq_length + pred_horizon data points
        if len(group_data) < seq_length + pred_horizon:
            continue
            
        # Stride can be 1 or larger to reduce data size
        stride = 24 
        
        for i in range(0, len(group_data) - seq_length - pred_horizon + 1, stride):
            x_seq = group_data[i : i + seq_length]
            y_seq = group_data[i + seq_length : i + seq_length + pred_horizon]
            
            X_all.append(x_seq)
            y_all.append(y_seq)
            
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # Reshape X to [samples, seq_length, features]
    # Currently features=1 (consumption only). 
    # TODO: Add external features (temp, wind) later if this works.
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))
    
    print(f"Data shape: X={X_all.shape}, y={y_all.shape}")
    
    return X_all, y_all, scaler

def train_lstm():
    from data_loader import load_data
    
    # Load data
    data = load_data('Dataset/20251111_JUNCTION_training.xlsx')
    df_cons = data['training_consumption']
    
    # Params
    SEQ_LENGTH = 168 # 1 week input
    PRED_HORIZON = 48 # 48h output
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    CHECKPOINT_DIR = 'models/checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Prepare data
    X, y, scaler = prepare_data(df_cons, SEQ_LENGTH, PRED_HORIZON)
    
    # Split Train/Val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False) 
    
    train_dataset = FortumDataset(X_train, y_train)
    val_dataset = FortumDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=PRED_HORIZON).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Resume logic
    start_epoch = 0
    latest_checkpoint = None
    
    # Find latest checkpoint
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('lstm_epoch_') and f.endswith('.pth')]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        
        # Check if params match (simple check)
        if checkpoint.get('hidden_size') == HIDDEN_SIZE and checkpoint.get('num_layers') == NUM_LAYERS:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed successfully. Starting from epoch {start_epoch+1}")
        else:
            print("Checkpoint parameters do not match current configuration. Starting from scratch.")
    
    # Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save Checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'lstm_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
    # Save final model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/lstm_model.pth')
    print("Model saved to models/lstm_model.pth")
    
    return model, scaler

if __name__ == "__main__":
    train_lstm()
