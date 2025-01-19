import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta

# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)


# Preprocess the data
df['date'] = pd.to_datetime(df['date'])
le_client = LabelEncoder()
le_action = LabelEncoder()
le_wkn = LabelEncoder()
df['client_id_encoded'] = le_client.fit_transform(df['client_id'])
df['action_encoded'] = le_action.fit_transform(df['action'])
df['wkn_encoded'] = le_wkn.fit_transform(df['wkn'])
df.sort_values(by=['client_id_encoded', 'date'], inplace=True)

# Create target variable
df['target'] = 0
for i in range(len(df)):
    client = df.iloc[i]['client_id_encoded']
    current_date = df.iloc[i]['date']
    buy_df = df[(df['client_id_encoded'] == client) & 
                (df['date'] > current_date) & 
                (df['date'] <= current_date + timedelta(days=30)) & 
                (df['action'] == 'buy')]
    if not buy_df.empty:
        df.at[i, 'target'] = 1

# Prepare sequences
grouped = df.groupby('client_id_encoded')
sequences = []
targets = []
client_ids = []

sequence_length = 50
for client_id, group in grouped:
    features = group[['action_encoded', 'wkn_encoded']].values
    y = group['target'].values
    for i in range(len(features) - sequence_length):
        seq = features[i:i+sequence_length]
        label = y[i+sequence_length]
        sequences.append(seq)
        targets.append(label)
        client_ids.append(client_id)

# Convert to numpy arrays
X = np.array(sequences)
y = np.array(targets)
clients = np.array(client_ids)

# Split into train and test
train_size = int(0.8 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
clients_train = clients[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]
clients_test = clients[train_size:]

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
clients_train = torch.from_numpy(clients_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
clients_test = torch.from_numpy(clients_test).long()

# Define LSTM model with client embeddings
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_clients, embedding_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.client_embedding = nn.Embedding(num_clients, embedding_dim)
        self.client_proj = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, clients):
        client_embed = self.client_embedding(clients)
        h_0 = self.client_proj(client_embed).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        out, (hn, cn) = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

input_dim = 2
hidden_dim = 50
output_dim = 1
num_clients = len(le_client.classes_)
embedding_dim = 10

model = LSTMModel(input_dim, hidden_dim, output_dim, num_clients, embedding_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader
train_dataset = TensorDataset(X_train, clients_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for sequences, clients, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences, clients)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item():.5f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test, clients_test)
    predicted = (outputs.squeeze() > 0.5).float()
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy:.4f}')

# Sample predictions
sample_indices = np.random.choice(len(X_test), size=5, replace=False)
for idx in sample_indices:
    print(f'Predicted: {predicted[idx].item()}, Actual: {y_test[idx].item()}')
