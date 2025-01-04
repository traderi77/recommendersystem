import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

# Assuming your DataFrame is named df and has columns: user_id, item_id, buy
# Example:
# df = pd.DataFrame({
#     'user_id': [1, 1, 2, 2, 3, 3],
#     'item_id': [1, 2, 1, 3, 2, 3],
#     'buy': [1, 0, 1, 1, 0, 1]
# })

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create mappings from user and item IDs to indices
user_ids = df['user_id'].unique()
item_ids = df['item_id'].unique()

user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

# Convert the train and test DataFrames to sparse matrices
def df_to_sparse_matrix(df, user_id_to_index, item_id_to_index):
    rows = df['user_id'].map(user_id_to_index)
    cols = df['item_id'].map(item_id_to_index)
    values = df['buy']
    return coo_matrix((values, (rows, cols)), shape=(len(user_id_to_index), len(item_id_to_index)))

train = df_to_sparse_matrix(train_df, user_id_to_index, item_id_to_index)
test = df_to_sparse_matrix(test_df, user_id_to_index, item_id_to_index)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# No need to filter interactions as per your request
filtered_train, filtered_test = train, test

print("Filtered train shape:", filtered_train.shape)
print("Filtered test shape:", filtered_test.shape)

# Train the LightFM model
model = LightFM(loss='bpr')
model.fit(filtered_train, epochs=50, num_threads=2)

# Evaluate the model
train_precision = precision_at_k(model, filtered_train, k=10).mean()
test_precision = precision_at_k(model, filtered_test, k=10).mean()

train_auc = auc_score(model, filtered_train).mean()
test_auc = auc_score(model, filtered_test).mean()

print(f"Train Precision@10: {train_precision:.4f}")
print(f"Test Precision@10: {test_precision:.4f}")
print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC: {test_auc:.4f}")
