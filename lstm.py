class TradePredictor(nn.Module):
    def __init__(self, num_clients, num_event_types, num_wkns, embedding_dim=64, lstm_dim=128):
        super().__init__()
        # Embeddings
        self.client_embedding = nn.Embedding(num_clients, embedding_dim)
        self.event_embedding = nn.Embedding(num_event_types, embedding_dim)
        self.wkn_embedding = nn.Embedding(num_wkns, embedding_dim)
        
        # Input features: event + wkn + client + time_delta + session_features
        input_dim = embedding_dim * 3 + 4  # 4 numerical features
        
        # LSTM with client-specific initialization
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_dim,
            batch_size=True,
            num_layers=2,  # Deeper network for more complex patterns
            dropout=0.1    # Prevent overfitting
        )
        
        # Prediction layers
        self.attention = nn.Linear(lstm_dim, 1)  # Attention over sequence
        self.predictor = nn.Linear(lstm_dim, 1)
        
    def forward(self, batch):
        # Unpack batch
        client_ids = batch['client_ids']
        event_types = batch['event_types']
        wkns = batch['wkns']
        time_deltas = batch['time_deltas']
        session_features = batch['session_features']  # [events_in_session, position_in_session, etc.]
        
        # Get embeddings
        client_embeds = self.client_embedding(client_ids)
        event_embeds = self.event_embedding(event_types)
        wkn_embeds = self.wkn_embedding(wkns)
        
        # Combine features
        inputs = torch.cat([
            client_embeds,
            event_embeds,
            wkn_embeds,
            time_deltas.unsqueeze(-1),
            session_features
        ], dim=-1)
        
        # Process sequence
        lstm_out, _ = self.lstm(inputs)
        
        # Attention over sequence
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        pred = torch.sigmoid(self.predictor(attended))
        return pred

def prepare_sequence(events, sequence_length=50):
    """
    Prepare a sequence of events for the model.
    """
    # Sort events by time
    events = sorted(events, key=lambda x: x['timestamp'])
    
    # Calculate session features
    current_session = None
    session_events = []
    session_features = []
    
    for i, event in enumerate(events):
        # New session if time gap > 30min
        if (current_session is None or 
            (event['timestamp'] - events[i-1]['timestamp']).minutes > 30):
            current_session = {
                'start_time': event['timestamp'],
                'events': [],
                'wkns': set()
            }
            session_events = []
        
        session_events.append(event)
        current_session['wkns'].add(event['wkn'])
        
        session_features.append({
            'events_in_session': len(session_events),
            'wkns_in_session': len(current_session['wkns']),
            'position_in_session': len(session_events),
            'session_duration': (event['timestamp'] - current_session['start_time']).minutes
        })
    
    return {
        'client_ids': [events[0]['client_id']] * len(events),  # Same client for sequence
        'event_types': [e['event_type'] for e in events],
        'wkns': [e['wkn'] for e in events],
        'time_deltas': [0] + [(events[i]['timestamp'] - events[i-1]['timestamp']).minutes 
                             for i in range(1, len(events))],
        'session_features': session_features
    }

def train_model(model, train_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Prepare sequences including session info
            sequences = [prepare_sequence(seq) for seq in batch['sequences']]
            
            # Get predictions
            preds = model(sequences)
            
            # Binary cross entropy loss
            loss = F.binary_cross_entropy(preds, batch['labels'])
            
            loss.backward()
            optimizer.step()
