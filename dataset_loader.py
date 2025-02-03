###
### You can do this without running this script
### This script filters the dataset for Python only
### Look at the dataset to see what you want
###

import dask_cudf
import config

# Load dataset across multiple GPUs
df = dask_cudf.read_parquet(config.DATASET_PATH)

# Display dataset structure
print(df.head())

Filter dataset for Python
df_filtered = df[df['language'].isin('Python')]

# Filter dataset for Python & JavaScript
# df_filtered = df[df['language'].isin(['Python', 'JavaScript'])]

# Save filtered dataset for training
df_filtered.to_parquet("filtered_stack.parquet")