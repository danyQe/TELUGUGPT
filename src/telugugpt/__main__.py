import os
from datasets import load_dataset
from torch.utils.data import random_split

from .config.config import MODEL_CONFIG, TRAINING_CONFIG, PATHS, DEVICE
from .models.transformer import build_transformer
from .data.dataset import TranslationDataset, create_dataloaders
from .tokenizers.tokenizer import create_tokenizers, load_tokenizers
from .utils.training import train_model

def main():
    # Create necessary directories
    os.makedirs(PATHS["checkpoints"], exist_ok=True)
    os.makedirs(os.path.dirname(PATHS["tokenizer_en"]), exist_ok=True)
    os.makedirs(os.path.dirname(PATHS["tokenizer_te"]), exist_ok=True)

    # Load dataset
    dataset = load_dataset(PATHS["dataset"], split="train")
    
    # Split dataset
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])
    
    # Take a subset for faster training during development
    train_subset_size = 1500
    val_subset_size = 50
    train_subset, _ = random_split(train_dataset, [train_subset_size, len(train_dataset) - train_subset_size])
    val_subset, _ = random_split(validation_dataset, [val_subset_size, len(validation_dataset) - val_subset_size])

    # Create or load tokenizers
    if not (os.path.exists(PATHS["tokenizer_en"]) and os.path.exists(PATHS["tokenizer_te"])):
        tokenizer_en, tokenizer_te = create_tokenizers(
            train_subset,
            ["instruction", "input", "output"],
            ["telugu_instruction", "telugu_input", "telugu_output"],
            PATHS["tokenizer_en"],
            PATHS["tokenizer_te"]
        )
    else:
        tokenizer_en, tokenizer_te = load_tokenizers(PATHS["tokenizer_en"], PATHS["tokenizer_te"])

    # Update max sequence length in config
    MODEL_CONFIG["max_seq_len"] = 512  # Or calculate from data

    # Create datasets
    train_ds = TranslationDataset(train_subset, MODEL_CONFIG["max_seq_len"], tokenizer_en, tokenizer_te)
    val_ds = TranslationDataset(val_subset, MODEL_CONFIG["max_seq_len"], tokenizer_en, tokenizer_te)

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(train_ds, val_ds, TRAINING_CONFIG["batch_size"])

    # Build model
    model = build_transformer(
        tokenizer_en.get_vocab_size(),
        tokenizer_te.get_vocab_size(),
        MODEL_CONFIG["max_seq_len"],
        MODEL_CONFIG["max_seq_len"],
        MODEL_CONFIG["d_model"],
        MODEL_CONFIG["num_blocks"],
        MODEL_CONFIG["num_heads"],
        MODEL_CONFIG["dropout_rate"],
        MODEL_CONFIG["d_ff"]
    ).to(DEVICE)

    # Train model
    train_model(
        model,
        train_dataloader,
        val_dataloader,
        tokenizer_en,
        tokenizer_te,
        PATHS["checkpoints"]
    )

if __name__ == "__main__":
    main() 