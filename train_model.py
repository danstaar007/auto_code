import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from dask_ml.model_selection import train_test_split, KFold
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import argparse
import config
import os

# Initialize TensorBoard
os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)
writer = SummaryWriter(config.TENSORBOARD_LOG_DIR)

# CLI Argument Parsing
parser = argparse.ArgumentParser(
    description="Train a GPT-2 model with various training configurations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--ddp", 
    action="store_true", 
    help="Enable Distributed Data Parallel (DDP) for multi-GPU training."
)

parser.add_argument(
    "--fsdp", 
    action="store_true", 
    help="Enable Fully Sharded Data Parallel (FSDP) for memory-efficient multi-GPU training."
)

parser.add_argument(
    "--cpu", 
    action="store_true", 
    help="Enable CPU Offloading (use Ryzen 7950X3D & Threadripper for hybrid training)."
)

parser.add_argument(
    "--optimizer", 
    choices=["adamw", "adam", "sgd"], 
    default=config.OPTIMIZER, 
    help="Choose the optimizer to use for training."
)

parser.add_argument(
    "--epochs", 
    type=int, 
    default=config.TRAIN_EPOCHS, 
    help="Number of training epochs."
)

parser.add_argument(
    "--batch_size", 
    type=int, 
    default=config.BATCH_SIZE, 
    help="Per-GPU batch size."
)

parser.add_argument(
    "--lr", 
    type=float, 
    default=config.LEARNING_RATE, 
    help="Set the learning rate for training."
)

parser.add_argument(
    "--kfold", 
    type=int, 
    default=config.KFOLD_SPLITS if config.USE_KFOLD else 0, 
    help="Enable K-Fold Cross-Validation with the specified number of folds (0 to disable)."
)

parser.add_argument(
    "--freeze", 
    action="store_true", 
    help="Enable scheduled layer freezing during training."
)

parser.add_argument(
    "--fsdp_shard_strategy", 
    choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"], 
    default=config.FSDP_SHARD_STRATEGY, 
    help="Choose FSDP sharding strategy."
)

parser.add_argument(
    "--fp16", 
    action="store_true", 
    default=config.FSDP_MIXED_PRECISION, 
    help="Enable mixed precision training (recommended for FSDP)."
)

parser.add_argument(
    "--tensorboard", 
    action="store_true", 
    help="Enable TensorBoard logging."
)

args = parser.parse_args()

# Initialize DDP or FSDP
if args.ddp or args.fsdp:
    init_process_group(backend="nccl")

# K-Fold Cross-Validation
df = load_dataset("parquet", data_files=config.DATASET_PATH, split="train")
if args.kfold > 1:
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    dataset_splits = list(kf.split(df))
else:
    train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
    dataset_splits = [(train_data, test_data)]

# Tokenization
tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True, max_length=config.MAX_SEQ_LENGTH)

# Train each fold
for fold, (train_idx, test_idx) in enumerate(dataset_splits):
    print(f"\n🔄 Training Fold {fold+1}/{args.kfold if args.kfold > 1 else 1}")

    # Prepare train/test datasets
    train_dataset = df.select(train_idx).map(tokenize_function, batched=True)
    test_dataset = df.select(test_idx).map(tokenize_function, batched=True)

    # Load model
    model = GPT2LMHeadModel.from_pretrained(config.MODEL_NAME)

    # Apply FSDP If Enabled
    if args.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.shard_strategy import ShardStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        shard_strategy = {
            "FULL_SHARD": ShardStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardStrategy.NO_SHARD,
        }.get(args.fsdp_shard_strategy, ShardStrategy.FULL_SHARD)

        model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrap_policy if config.FSDP_AUTO_WRAP_POLICY == "TRANSFORMER_BASED" else None,
            mixed_precision=args.fp16,
            sharding_strategy=shard_strategy,
        )
        print(f"✅ Using FSDP with `{args.fsdp_shard_strategy}` sharding strategy.")

    # Apply CPU Offloading If Needed
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    # Select Optimizer
    optimizer = {
        "adamw": AdamW,
        "adam": torch.optim.Adam,
        "sgd": SGD,
    }[args.optimizer](model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config.CHECKPOINT_PATH,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.TENSORBOARD_LOG_DIR if args.tensorboard else None,
        logging_steps=100,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
    )

    # Trainer Setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        optimizers=(optimizer, None),
    )

    trainer.train()

    trainer.save_model(config.CHECKPOINT_PATH)
    tokenizer.save_pretrained(config.CHECKPOINT_PATH)

writer.close()