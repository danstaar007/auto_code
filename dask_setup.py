from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import config

# Assign GPU visibility based on machine setup
if "192.168.1.10" in config.MACHINE_IPS:
    CUDA_VISIBLE_DEVICES = [0, 1] 
elif "192.168.1.11" in config.MACHINE_IPS:
    CUDA_VISIBLE_DEVICES = [0] 

# Start Dask-CUDA cluster
cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES)
client = Client(cluster)
print(client)