
# 2. Create sequences
sequences = create_sequences(df, sequence_length=3)

# 3. Create vocabulary mappings for embeddings
client_to_idx = {client: idx for idx, client in enumerate(df['client_id'].unique())}
action_to_idx = {action: idx for idx, action in enumerate(df['action'].unique())}

# 4. Initialize model
model = TradePredictor(
    num_clients=len(client_to_idx),
    num_actions=len(action_to_idx),
    num_wkns=len(),
    embedding_dim=64,
    lstm_dim=128
)

# 5. Create DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(sequences, batch_size=32, shuffle=True)

# 6. Train
optimizer = torch.optim.Adam(model.parameters())
train_model(model, train_loader, optimizer, epochs=10)

