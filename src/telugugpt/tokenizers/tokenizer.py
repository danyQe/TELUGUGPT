from tokenizers import Tokenizer, BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from itertools import chain
from typing import List, Iterator, Optional
import os

def get_ds_iterator(raw_dataset, field: str) -> Iterator[str]:
    """Iterator to get text from dataset field."""
    for data in raw_dataset:
        if data[field] is not None:
            yield data[field]

def train_tokenizer(raw_train_dataset,
                   fields: List[str],
                   save_path: str,
                   vocab_size: Optional[int] = 30000) -> Tokenizer:
    """Train a BPE tokenizer on the given dataset fields.
    
    Args:
        raw_train_dataset: Dataset containing text fields
        fields: List of field names to use for training
        save_path: Path to save the trained tokenizer
        vocab_size: Maximum vocabulary size (default: 30000)
    
    Returns:
        Trained tokenizer
    """
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.pre_tokenizer = Whitespace()

    # Chain all text fields into a single iterator
    texts = chain(*[get_ds_iterator(raw_train_dataset, field) for field in fields])
    
    # Train the tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # Save the tokenizer
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    
    return tokenizer

def create_tokenizers(raw_train_dataset,
                     english_fields: List[str],
                     telugu_fields: List[str],
                     en_save_path: str,
                     te_save_path: str,
                     en_vocab_size: int = 30000,
                     te_vocab_size: int = 30000) -> tuple[Tokenizer, Tokenizer]:
    """Create and train tokenizers for both English and Telugu.
    
    Args:
        raw_train_dataset: Dataset containing both English and Telugu text
        english_fields: List of English field names
        telugu_fields: List of Telugu field names
        en_save_path: Path to save English tokenizer
        te_save_path: Path to save Telugu tokenizer
        en_vocab_size: Maximum vocabulary size for English (default: 30000)
        te_vocab_size: Maximum vocabulary size for Telugu (default: 30000)
    
    Returns:
        Tuple of (English tokenizer, Telugu tokenizer)
    """
    
    # Train English tokenizer
    tokenizer_en = train_tokenizer(
        raw_train_dataset,
        english_fields,
        en_save_path,
        vocab_size=en_vocab_size
    )
    
    # Train Telugu tokenizer
    tokenizer_te = train_tokenizer(
        raw_train_dataset,
        telugu_fields,
        te_save_path,
        vocab_size=te_vocab_size
    )
    
    return tokenizer_en, tokenizer_te

def load_tokenizers(en_path: str, te_path: str) -> tuple[Tokenizer, Tokenizer]:
    """Load pre-trained tokenizers.
    
    Args:
        en_path: Path to English tokenizer
        te_path: Path to Telugu tokenizer
    
    Returns:
        Tuple of (English tokenizer, Telugu tokenizer)
    """
    tokenizer_en = Tokenizer.from_file(en_path)
    tokenizer_te = Tokenizer.from_file(te_path)
    return tokenizer_en, tokenizer_te 