import torch

# Model Configuration
MODEL_CONFIG = {
    "d_model": 512,
    "num_blocks": 6,
    "num_heads": 8,
    "dropout_rate": 0.1,
    "d_ff": 2048,
    "max_seq_len": None  # Will be set during runtime
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 2,
    "learning_rate": 1e-4,
    "epochs": 10,
    "gradient_accumulation_steps": 4,
    "label_smoothing": 0.1
}

# Paths Configuration
PATHS = {
    "tokenizer_en": "./Tokenizer_en/tokenizer_en.json",
    "tokenizer_te": "./Tokenizer_te/tokenizer_te.json",
    "checkpoints": "./TeluguGPT",
    "dataset": "Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized"
}

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 