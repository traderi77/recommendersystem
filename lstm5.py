import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple

class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_length: int = 10, prediction_window: int = 30):
        self.df = df.copy()
        # Ensure date is datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        
        # Encode categorical variables
        self.client_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        self.wkn_encoder = LabelEncoder()
        
        self.df['client_id_encoded'] = self.client_encoder.fit_transform(df['client_id'])
        self.df['action_encoded'] = self.action_encoder.fit_transform(df['action'])
        self.df['wkn_encoded'] = self.wkn_encoder.fit_transform(df['wkn'])
        
        # Create sequences
        self.sequences = self._create_sequences()

    def _create_sequences(self) -> List[Tuple]:
        sequences = []
        for client_id in self.df['client_id'].unique():
            client_data = self.df[self.df['client_id'] == client_id]
            
            for i in range(len(client_data) - self.sequence_length):
                # Get sequence window
                sequence = client_data.iloc[i:i + self.sequence_length]
                sequence_end_date = sequence.iloc[-1]['date']
                
                # Get target window (next 30 days)
                target_window = client_data[
                    (client_data['date'] > sequence_end_date) & 
                    (client_data['date'] <= sequence_end_date + pd.Timedelta(days=self.prediction_window))
                ]
                
                # Check if there's a BUY action for the last WKN in sequence
                last_wkn = sequence.iloc[-1]['wkn']
                target = int(any((target_window['action'] == 'BUY') & 
                               (target_window['wkn'] == last_wkn)))
                
                # Calculate time differences in days
                sequence_times = (sequence['date'] - sequence.iloc[0]['date']).dt.total_seconds() / (24*3600)
                time_tensor = torch.tensor(sequence_times.values, dtype=torch.float)
                
                # Create sequence tensors
                client_tensor = torch.tensor(sequence['client_id_encoded'].values[0], dtype=torch.long)
                action_tensor = torch.tensor(sequence['action_encoded'].values, dtype=torch.long)
                wkn_tensor = torch.tensor(sequence['wkn_encoded'].values, dtype=torch.long)
                
                sequences.append((client_tensor, action_tensor, wkn_tensor, time_tensor, target))
        
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

class ClientAwareSequenceModel(nn.Module):
    def __init__(self, 
                 num_clients: int,
                 num_actions: int,
                 num_wkns: int,
                 embedding_dim: int = 32,
                 hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        
        # Embeddings
        self.client_embedding = nn.Embedding(num_clients, embedding_dim)
        self.action_embedding = nn.Embedding(num_actions, embedding_dim)
        self.wkn_embedding = nn.Embedding(num_wkns, embedding_dim)
        
        # Time encoding layer
        self.time_encoding = nn.Linear(1, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 4,  # Combined embeddings + time
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Output layers with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, client, actions, wkns, times):
        # Get embeddings
        client_emb = self.client_embedding(client).unsqueeze(1).repeat(1, actions.size(1), 1)
        action_emb = self.action_embedding(actions)
        wkn_emb = self.wkn_embedding(wkns)
        time_emb = self.time_encoding(times.unsqueeze(-1))
        
        # Combine embeddings
        combined = torch.cat([client_emb, action_emb, wkn_emb, time_emb], dim=2)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(combined)
        
        # Use last LSTM output
        last_out = lstm_out[:, -1, :]
        
        # Pass through final layers
        x = self.dropout(last_out)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        
        return x.squeeze()

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for client, actions, wkns, times, targets in train_loader:
            # Move to device
            client = client.to(device)
            actions = actions.to(device)
            wkns = wkns.to(device)
            times = times.to(device)
            targets = targets.float().to(device)
            
            # Forward pass
            outputs = model(client, actions, wkns, times)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for client, actions, wkns, times, targets in val_loader:
                client = client.to(device)
                actions = actions.to(device)
                wkns = wkns.to(device)
                times = times.to(device)
                targets = targets.float().to(device)
                
                outputs = model(client, actions, wkns, times)
                val_loss += criterion(outputs, targets).item()
                
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        print(f'Epoch {epoch + 1}')
        print(f'Training Loss: {train_loss / len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%\n')








dataset = SequenceDataset(df, sequence_length=5, prediction_window=30)

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model
model = ClientAwareSequenceModel(
    num_clients=len(dataset.client_encoder.classes_),
    num_actions=len(dataset.action_encoder.classes_),
    num_wkns=len(dataset.wkn_encoder.classes_),
    embedding_dim=16,  
    hidden_dim=32,
)

train_model(model, train_loader, val_loader, num_epochs=20)

# Example prediction
def make_prediction(model, dataset, client_id, wkn, sequence_length=5):
    # Get recent interactions for this client-wkn pair
    client_wkn_data = df[
        (df['client_id'] == client_id) & 
        (df['wkn'] == wkn)
    ].tail(sequence_length)
    
    if len(client_wkn_data) < sequence_length:
        return None  # Not enough data
    
    # Prepare input tensors
    client_tensor = torch.tensor(
        dataset.client_encoder.transform([client_id])[0], 
        dtype=torch.long
    )
    action_tensor = torch.tensor(
        dataset.action_encoder.transform(client_wkn_data['action'].values), 
        dtype=torch.long
    )
    wkn_tensor = torch.tensor(
        dataset.wkn_encoder.transform(client_wkn_data['wkn'].values), 
        dtype=torch.long
    )
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prob = model(
            client_tensor.unsqueeze(0),
            action_tensor.unsqueeze(0),
            wkn_tensor.unsqueeze(0)
        )
    
    return float(prob)

# Example predictions for a few client-wkn pairs
print("\nSample predictions:")
for _ in range(3):
    client_id = random.choice(df['client_id'].unique())
    wkn = random.choice(df[df['client_id'] == client_id]['wkn'].unique())
    prob = make_prediction(model, dataset, client_id, wkn)
    print(f"Client {client_id}, WKN {wkn}: {prob:.3f} probability of purchase")
