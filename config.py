vocab_size = 32000  # Number of unique tokens in the vocabulary (overridden by trained tokenizer if different)
max_seq_len = 128  # Maximum sequence length
d_model = 128  # Transformer hidden size / embedding dimension
n_heads = 4  # Size of attention heads in multi-head self-attention
n_layers = 6  # Size of Transformer blocks
d_ff = 512  # Hidden dimension of the FFN
dropout = 0.1  # Dropout probability applied during training
batch_size = 8  # Training batch size
lr = 1e-4  # Learning rate for the optimizer.
weight_decay = 0.02  # weight decay
epochs = 3  # Week 5–6: 2–3 epochs on a single GPU
max_steps_per_epoch = 10000  # cap steps per epoch (keeps training manageable)
warmup_steps = 1000  # Warmup steps for the learning-rate schedule
grad_accum_steps = 1  # Gradient accumulation steps to simulate a larger batch
device = "cuda"  # fallback; train.py uses cuda if available

# Logging, checkpoint, plots
log_every = 50  # train loss every N optimizer steps
eval_every = 500  # validation loss every N optimizer steps
checkpoint_every = 500  # save checkpoint every N optimizer steps
checkpoint_dir = "checkpoints"
learning_curve_path = "learning_curve.png"
final_model_path = "gpt_model.pt"
tokenizer_path = "tokenizer/trained_tokenizer/tokenizer.json"
grad_clip = 1.0
num_workers = 0  # dataloader workers (0 is safest on macOS)
# If set, validation loss will be computed on only the first N batches.
# This prevents training from "hanging" during full validation sweeps.
val_max_batches = 50
