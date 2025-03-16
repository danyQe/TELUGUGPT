# TeluguGPT

TeluguGPT is a state-of-the-art neural machine translation model that translates English text to Telugu (తెలుగు), a Dravidian language primarily spoken in southern India. The model implements the complete Transformer architecture as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Features

- Complete implementation of the Transformer architecture
- Custom BPE tokenization for both English and Telugu languages
- Efficient data processing with dynamic batching and padding
- Gradient accumulation for training with limited resources
- Checkpoint saving and loading for interrupted training
- Comprehensive validation with example translations
- Type hints and documentation for better code understanding

## Installation

1. Clone the repository:
```bash
git clone https://github.com/danyQe/TeluguGPT.git
cd TeluguGPT
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
├── telugugpt/
│   ├── __init__.py           # Package initialization
│   ├── __main__.py           # Main execution file
│   ├── config/
│   │   └── config.py         # Configuration settings
│   ├── data/
│   │   └── dataset.py        # Dataset handling
│   ├── models/
│   │   ├── layers.py         # Model layers
│   │   └── transformer.py    # Transformer architecture
│   ├── tokenizers/
│   │   └── tokenizer.py      # Tokenizer handling
│   └── utils/
│       └── training.py       # Training utilities
```

## Usage

### Training

1. Configure the model parameters in `src/telugugpt/config/config.py`:
```python
MODEL_CONFIG = {
    "d_model": 512,
    "num_blocks": 6,
    "num_heads": 8,
    "dropout_rate": 0.1,
    "d_ff": 2048
}
```

2. Start training:
```bash
python -m src.telugugpt
```

### Translation

```python
from telugugpt import load_tokenizers, build_transformer
from telugugpt.config import MODEL_CONFIG, PATHS, DEVICE

# Load tokenizers
tokenizer_en, tokenizer_te = load_tokenizers(PATHS["tokenizer_en"], PATHS["tokenizer_te"])

# Load model
model = build_transformer(
    tokenizer_en.get_vocab_size(),
    tokenizer_te.get_vocab_size(),
    MODEL_CONFIG["max_seq_len"],
    MODEL_CONFIG["max_seq_len"]
).to(DEVICE)

# Load checkpoint
checkpoint = torch.load("path/to/checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Translate
text = "How are you?"
translated = model.translate(text)
print(translated)  # Outputs Telugu translation
```

## Model Architecture

### Transformer Components
- **Encoder**: 6 layers of self-attention and feed-forward networks
- **Decoder**: 6 layers with masked self-attention, cross-attention, and feed-forward networks
- **Attention Heads**: 8 heads per layer
- **Model Dimension**: 512
- **Feed-Forward Dimension**: 2048
- **Dropout Rate**: 0.1

### Training Parameters
- **Batch Size**: 2
- **Learning Rate**: 1e-4
- **Epochs**: 10
- **Gradient Accumulation Steps**: 4
- **Label Smoothing**: 0.1

## Training

### Dataset
The model is trained on the [Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized](https://huggingface.co/datasets/Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized) dataset, which contains:
- English instructions/text
- Telugu translations
- Additional metadata and filtering

### Training Process
1. **Data Preprocessing**:
   - Text tokenization using BPE
   - Dynamic sequence padding
   - Mask generation for attention

2. **Training Loop**:
   - Gradient accumulation for effective batch size
   - Label smoothing for better generalization
   - Regular checkpointing
   - Validation with example translations

3. **Monitoring**:
   - Loss tracking
   - Translation quality assessment
   - Model parameter statistics

### Checkpoints
The model saves checkpoints after each epoch in the `./TeluguGPT/` directory with the format `model_{epoch}.pt`. Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Current epoch
- Global step

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Implementation based on ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.
- Dataset provided by [Telugu-LLM-Labs](https://huggingface.co/Telugu-LLM-Labs)
- Thanks to the PyTorch team for their excellent deep learning framework
