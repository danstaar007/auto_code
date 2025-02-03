import dask_cudf
import config

# Load dataset (only metadata first)
df = dask_cudf.read_parquet(config.DATASET_PATH, columns=['content'])

# Estimate size (GB)
estimated_size_gb = df.memory_usage(deep=True).sum().compute() / 1e9
print(f"\n📊 Estimated Dataset Size: {estimated_size_gb:.2f} GB\n")

# Check GPU capacity
gpu_vram_total_gb = 12 * config.TOTAL_GPUS  # 12GB per 4070 Ti
if estimated_size_gb > gpu_vram_total_gb:
    print(f"⚠️ Dataset is larger than GPU VRAM ({gpu_vram_total_gb} GB). Consider using CPU offloading or mixed precision training.")
else:
    print("✅ Dataset should fit into GPU VRAM. Proceeding with training.")