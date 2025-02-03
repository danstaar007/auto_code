NUM_GPUS_PER_MACHINE = [2, 1]  # [GPUs on Machine 1, GPUs on Machine 2]
TOTAL_GPUS = sum(NUM_GPUS_PER_MACHINE)
MACHINE_IPS = ["192.168.1.10", "192.168.1.11"]

DATASET_PATH = "filtered_stack.parquet"
MODEL_NAME = "gpt2"
TRAIN_EPOCHS = 10
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 5e-5
OPTIMIZER = "adamw"  # Optimizer options: "adamw", "adam", "sgd"
WEIGHT_DECAY = 0.01

# Layer Freezing (Set epochs to progressively unfreeze)
FREEZE_LAYERS = True
FREEZE_SCHEDULE = {0: "all", 3: "half", 6: "none"}  # Example: Freeze all for 3 epochs, half for 3, then unfreeze

# K-Fold Cross-Validation
USE_KFOLD = True
KFOLD_SPLITS = 5

CHECKPOINT_PATH = "./checkpoints"
EARLY_STOPPING_PATIENCE = 3
LOSS_CONVERGENCE_THRESHOLD = 0.01
API_PORT = 5000  # API Port for code completion
HOTKEY = "ctrl+shift+c"  # Hotkey for code completion
TENSORBOARD_LOG_DIR = "./tensorboard_logs"

### FSDP-SPECIFIC CONFIGURATIONS ###
USE_FSDP = False  # Set to True if using FSDP (Automatically enabled with `--fsdp` in CLI)
FSDP_SHARD_STRATEGY = "FULL_SHARD"  # Options: "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"
FSDP_AUTO_WRAP_POLICY = "TRANSFORMER_BASED"  # Options: "TRANSFORMER_BASED", "LAYER_BASED"
FSDP_MIXED_PRECISION = True  # Enable mixed precision (recommended for lower memory usage)
FSDP_FP16_REDUCE_SCATTER = True  # Reduce-scatter optimization for lower memory usage