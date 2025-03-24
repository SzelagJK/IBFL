import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Shuffle
df = pd.read_csv('diabetes_data.csv')
shuffled_df = df.sample(frac=1).reset_index(drop=True)

# Normalise
binary_columns = shuffled_df.columns[shuffled_df.isin([0, 1]).all()].tolist()
non_binary_columns = [col for col in shuffled_df.columns if col not in binary_columns]
scaler = StandardScaler()
shuffled_df[non_binary_columns] = scaler.fit_transform(shuffled_df[non_binary_columns])

# Split
split_21, split_19 = train_test_split(shuffled_df, test_size=0.475, random_state=42)
split_21.to_csv('split_21.csv', index=False)
split_19.to_csv('split_19.csv', index=False)