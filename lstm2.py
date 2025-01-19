def create_sequences(df, sequence_length=50):
    """
    Create sequences from a dataframe with columns: wkn, session_id, action, date
    Returns sequences suitable for training the LSTM
    """
    # Sort by date first
    df = df.sort_values('date')
    
    sequences = []
    
    # Group by client
    for client_id, client_df in df.groupby('client_id'):
        client_events = client_df.values.tolist()  # [wkn, session_id, action, date]
        bought_wkns = set()
        
        # Sliding window over events
        for i in range(len(client_events) - sequence_length):
            history = client_events[i:i+sequence_length]
            current_date = history[-1][3]  # date of last event in history
            
            # Find events in next 30 days for target
            future_window = [
                e for e in client_events[i+sequence_length:] 
                if (e[3] - current_date).days <= 30
            ]
            
            # Get WKNs interacted with in history (excluding already bought)
            candidate_wkns = {e[0] for e in history if e[0] not in bought_wkns}
            
            # Update bought WKNs
            bought_wkns.update(e[0] for e in history if e[2] == 'buy')
            
            # Get purchased WKNs in future window
            purchased_wkns = {e[0] for e in future_window if e[2] == 'buy'}
            
            # Create sequence for each candidate WKN
            for wkn in candidate_wkns:
                sequences.append({
                    'sequence': history,
                    'client_id': client_id,
                    'wkn': wkn,
                    'session_ids': [e[1] for e in history],
                    'time_deltas': [0] + [
                        (history[j+1][3] - history[j][3]).total_seconds() / 3600  # hours
                        for j in range(len(history)-1)
                    ],
                    'label': 1 if wkn in purchased_wkns else 0
                })
    
    return sequences
