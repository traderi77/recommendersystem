import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict

def create_sequences(df, sequence_length=50):
    """
    Create sequences from a dataframe with columns: wkn, client_id, date, action
    Note: trades (action='buy') only have date (no timestamp)
    """
    # For trades, set time to end of day to ensure proper ordering
    df['date'] = df.apply(lambda row: 
        row['date'].replace(hour=23, minute=59, second=59) if row['action'] == 'buy' 
        else row['date'], axis=1)
    
    # Sort by date
    df = df.sort_values('date')
    
    sequences = []
    
    # Group by client
    for client_id, client_df in df.groupby('client_id'):
        client_events = client_df.values.tolist()  # [wkn, client_id, date, action]
        bought_wkns = set()
        
        # Sliding window over events
        for i in range(len(client_events) - sequence_length):
            history = client_events[i:i+sequence_length]
            current_date = history[-1][2].date()  # just the date part
            
            # Find events in next 30 days for target
            future_window = [
                e for e in client_events[i+sequence_length:] 
                if (e[2].date() - current_date).days <= 30
            ]
            
            # Get WKNs interacted with in history (excluding already bought)
            candidate_wkns = {e[0] for e in history if e[0] not in bought_wkns}
            
            # Update bought WKNs
            bought_wkns.update(e[0] for e in history if e[3] == 'buy')
            
            # Get purchased WKNs in future window
            purchased_wkns = {e[0] for e in future_window if e[3] == 'buy'}
            
            # Create sequence for each candidate WKN
            for wkn in candidate_wkns:
                sequences.append({
                    'sequence': history,
                    'client_id': client_id,
                    'wkn': wkn,
                    'time_deltas': [0] + [
                        (history[j+1][2] - history[j][2]).total_seconds() / 3600  # hours
                        for j in range(len(history)-1)
                    ],
                    'label': 1 if wkn in purchased_wkns else 0
                })
    
    return sequences

class TradePredictor(nn.Module):
    def __init__(self, num_clients, num_actions, num_wkns, embedding_dim=64, lstm_dim=128):
        super().__init__()
        # Embeddings
        self.client_embedding = nn.Embedding(num_clients, embedding_dim)
        self.action_embedding = nn.Embedding(num_actions, embedding_dim)
        self.wkn_embedding = nn.Embedding(num_wkns, embedding_dim)
        
        # Input features: action + wkn + client + time_delta
        input_dim = embedding_dim * 3 + 1  # +1 for time delta
        
        # LSTM with client-specific initialization
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_dim,
            batch_size=True,
            num_layers=2,
            dropout=0.1
        )
        
        # Attention and prediction layers
        self.attention = nn.Linear(lstm_dim, 1)
        self.predictor = nn.Linear(lstm_dim, 1)
        
    def forward(self, batch):
        # Unpack batch
        client_ids = batch['client_ids']
        actions = batch['actions']
        wkns = batch['wkns']
        time_deltas = batch['time_deltas']
        
        # Get embeddings
        client_embeds = self.client_embedding(client_ids)
        action_embeds = self.action_embedding(actions)
        wkn_embeds = self.wkn_embedding(wkns)
        
        # Combine features
        inputs = torch.cat([
            client_embeds,
            action_embeds,
            wkn_embeds,
            time_deltas.unsqueeze(-1)
        ], dim=-1)
        
        # Process sequence
        lstm_out, _ = self.lstm(inputs)
        
        # Attention over sequence
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        pred = torch.sigmoid(self.predictor(attended))
        return pred

def prepare_batch(sequences):
    """Convert sequence dictionaries into tensors for the model"""
    batch = {
        'client_ids': [],
        'actions': [],
        'wkns': [],
        'time_deltas': [],
        'labels': []
    }
    
    for seq in sequences:
        batch['client_ids'].append(seq['client_id'])
        batch['actions'].extend([e[3] for e in seq['sequence']])  # action
        batch['wkns'].extend([e[0] for e in seq['sequence']])    # wkn
        batch['time_deltas'].extend(seq['time_deltas'])
        batch['labels'].append(seq['label'])
    
    # Convert to tensors
    return {k: torch.tensor(v) for k, v in batch.items()}

def train_model(model, train_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_sequences in train_loader:
            optimizer.zero_grad()
            
            # Prepare batch
            batch = prepare_batch(batch_sequences)
            
            # Get predictions
            preds = model(batch)
            
            # Binary cross entropy loss
            loss = F.binary_cross_entropy(preds, batch['labels'])
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f'Epoch {epoch}: Loss = {epoch_loss/len(train_loader)}')

def predict(model, client_history, candidate_wkns):
    """Make predictions for a client's candidate WKNs"""
    model.eval()
    with torch.no_grad():
        sequences = []
        for wkn in candidate_wkns:
            seq = prepare_sequence(client_history, wkn)
            sequences.append(seq)
        
        batch = prepare_batch(sequences)
        predictions = model(batch)
        
        # Return WKN-probability pairs
        return list(zip(candidate_wkns, predictions.numpy()))
