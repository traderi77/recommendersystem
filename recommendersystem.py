import pandas as pd
import numpy as np
from collections import Counter
import itertools

class OttoRecommender:
    def __init__(self):
        self.type_weight_multipliers = {0: 1, 1: 6, 2: 3}  # clicks: 0, carts: 1, orders: 2
        
    def create_covisitation_matrices(self, df):
        """
        Creates all three covisitation matrices from a single DataFrame
        """
        # Get popular items from test set
        self.top_clicks = df[df['type']==0]['aid'].value_counts().index.values[:20]
        self.top_orders = df[df['type'].isin([1,2])]['aid'].value_counts().index.values[:20]
        
        # Create the three matrices
        self.clicks_matrix = self._create_time_weighted_matrix(df)
        self.carts_orders_matrix = self._create_type_weighted_matrix(df)
        self.buy2buy_matrix = self._create_buy2buy_matrix(df)
        
    def _create_type_weighted_matrix(self, df, n_items=15):
        """Creates type-weighted covisitation matrix for cart/order recommendations"""
        # Sort by session and timestamp
        df = df.sort_values(['session', 'ts'], ascending=[True, False])
        
        # Keep last 30 events per session
        df['n'] = df.groupby('session').cumcount()
        df = df[df.n < 30].drop('n', axis=1)
        
        # Create pairs within 24-hour window
        pairs = df.merge(df, on='session')
        pairs = pairs[
            (abs(pairs.ts_x - pairs.ts_y) < 24 * 60 * 60) &
            (pairs.aid_x != pairs.aid_y)
        ]
        
        # Weight by type and aggregate
        pairs = pairs[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
        pairs['wgt'] = pairs.type_y.map(self.type_weight_multipliers)
        
        # Get top n_items per aid_x
        agg = pairs.groupby(['aid_x', 'aid_y'])['wgt'].sum().reset_index()
        agg = agg.sort_values(['aid_x', 'wgt'], ascending=[True, False])
        agg['rank'] = agg.groupby('aid_x').cumcount()
        top_pairs = agg[agg.rank < n_items]
        
        return self._pairs_to_dict(top_pairs)
    
    def _create_buy2buy_matrix(self, df, n_items=15):
        """Creates covisitation matrix from cart/order to cart/order"""
        # Filter for cart and order events
        df = df[df['type'].isin([1, 2])]
        
        # Sort by session and timestamp
        df = df.sort_values(['session', 'ts'], ascending=[True, False])
        
        # Keep last 30 events per session
        df['n'] = df.groupby('session').cumcount()
        df = df[df.n < 30].drop('n', axis=1)
        
        # Create pairs within 14-day window
        pairs = df.merge(df, on='session')
        pairs = pairs[
            (abs(pairs.ts_x - pairs.ts_y) < 14 * 24 * 60 * 60) &
            (pairs.aid_x != pairs.aid_y)
        ]
        
        # Aggregate with weight=1
        pairs = pairs[['session', 'aid_x', 'aid_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
        pairs['wgt'] = 1
        
        # Get top n_items per aid_x
        agg = pairs.groupby(['aid_x', 'aid_y'])['wgt'].sum().reset_index()
        agg = agg.sort_values(['aid_x', 'wgt'], ascending=[True, False])
        agg['rank'] = agg.groupby('aid_x').cumcount()
        top_pairs = agg[agg.rank < n_items]
        
        return self._pairs_to_dict(top_pairs)
    
    def _create_time_weighted_matrix(self, df, n_items=20):
        """Creates time-weighted covisitation matrix for click recommendations"""
        START_TIME = 1659304800
        END_TIME = 1662328791
        
        # Sort by session and timestamp
        df = df.sort_values(['session', 'ts'], ascending=[True, False])
        
        # Keep last 30 events per session
        df['n'] = df.groupby('session').cumcount()
        df = df[df.n < 30].drop('n', axis=1)
        
        # Create pairs within 24-hour window
        pairs = df.merge(df, on='session')
        pairs = pairs[
            (abs(pairs.ts_x - pairs.ts_y) < 24 * 60 * 60) &
            (pairs.aid_x != pairs.aid_y)
        ]
        
        # Weight by time and aggregate
        pairs = pairs[['session', 'aid_x', 'aid_y', 'ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
        pairs['wgt'] = 1 + 3 * (pairs.ts_x - START_TIME) / (END_TIME - START_TIME)
        
        # Get top n_items per aid_x
        agg = pairs.groupby(['aid_x', 'aid_y'])['wgt'].sum().reset_index()
        agg = agg.sort_values(['aid_x', 'wgt'], ascending=[True, False])
        agg['rank'] = agg.groupby('aid_x').cumcount()
        top_pairs = agg[agg.rank < n_items]
        
        return self._pairs_to_dict(top_pairs)
    
    def _pairs_to_dict(self, df):
        """Converts DataFrame of pairs to dictionary format"""
        return df.groupby('aid_x').aid_y.apply(list).to_dict()
    
    def suggest_clicks(self, df):
        """Suggests next clicks for a user session"""
        # Get user history
        aids = df.aid.tolist()
        types = df.type.tolist()
        unique_aids = list(dict.fromkeys(aids[::-1]))
        
        # If enough history, use weighted approach
        if len(unique_aids) >= 20:
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * self.type_weight_multipliers[t]
            return [k for k, v in aids_temp.most_common(20)]
        
        # Otherwise use covisitation matrices
        aids2 = list(itertools.chain(*[self.clicks_matrix[aid] 
                                     for aid in unique_aids 
                                     if aid in self.clicks_matrix]))
        top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) 
                    if aid2 not in unique_aids]
        result = unique_aids + top_aids2[:20 - len(unique_aids)]
        
        # Fill remaining with popular items
        return result + list(self.top_clicks)[:20 - len(result)]
    
    def suggest_buys(self, df):
        """Suggests next cart/order items for a user session"""
        # Get user history
        aids = df.aid.tolist()
        types = df.type.tolist()
        unique_aids = list(dict.fromkeys(aids[::-1]))
        
        # Get buy history
        df_buys = df[df['type'].isin([1, 2])]
        unique_buys = list(dict.fromkeys(df_buys.aid.tolist()[::-1]))
        
        # If enough history, use weighted approach
        if len(unique_aids) >= 20:
            weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            
            # Weight by type and frequency
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * self.type_weight_multipliers[t]
            
            # Add buy2buy recommendations
            aids3 = list(itertools.chain(*[self.buy2buy_matrix[aid] 
                                         for aid in unique_buys 
                                         if aid in self.buy2buy_matrix]))
            for aid in aids3:
                aids_temp[aid] += 0.1
                
            return [k for k, v in aids_temp.most_common(20)]
        
        # Otherwise use covisitation matrices
        aids2 = list(itertools.chain(*[self.carts_orders_matrix[aid] 
                                     for aid in unique_aids 
                                     if aid in self.carts_orders_matrix]))
        aids3 = list(itertools.chain(*[self.buy2buy_matrix[aid] 
                                     for aid in unique_buys 
                                     if aid in self.buy2buy_matrix]))
        
        top_aids2 = [aid2 for aid2, cnt in Counter(aids2 + aids3).most_common(20) 
                    if aid2 not in unique_aids]
        result = unique_aids + top_aids2[:20 - len(unique_aids)]
        
        # Fill remaining with popular items
        return result + list(self.top_orders)[:20 - len(result)]
    
    def generate_recommendations(self, test_df):
        """
        Generate recommendations for all users in test_df
        Returns DataFrame with session_type and labels columns
        """
        # Generate predictions for clicks and buys
        pred_clicks = test_df.sort_values(['session', 'ts']).groupby(['session']).apply(
            lambda x: self.suggest_clicks(x))
        pred_buys = test_df.sort_values(['session', 'ts']).groupby(['session']).apply(
            lambda x: self.suggest_buys(x))
        
        # Create submission DataFrame
        clicks_df = pd.DataFrame(pred_clicks.add_suffix('_clicks'), 
                               columns=['labels']).reset_index()
        orders_df = pd.DataFrame(pred_buys.add_suffix('_orders'), 
                               columns=['labels']).reset_index()
        carts_df = pd.DataFrame(pred_buys.add_suffix('_carts'), 
                              columns=['labels']).reset_index()
        
        # Combine all predictions
        pred_df = pd.concat([clicks_df, orders_df, carts_df])
        pred_df.columns = ['session_type', 'labels']
        pred_df['labels'] = pred_df.labels.apply(lambda x: ' '.join(map(str, x)))
        
        return pred_df

# Example usage:
# recommender = OttoRecommender()
# recommender.create_covisitation_matrices(train_df)  # Train the model
# predictions = recommender.generate_recommendations(test_df)  # Generate predictions