import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from typing import Dict, Any

class TranslationDataset(Dataset):
    def __init__(self, raw_dataset, max_seq_len: int, tokenizer_en: Tokenizer, tokenizer_te: Tokenizer):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.max_seq_len = max_seq_len
        self.tokenizer_en = tokenizer_en
        self.tokenizer_te = tokenizer_te

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index) -> Dict[str, Any]:
        raw_text = self.raw_dataset[index]
        source_text = " ".join([str(raw_text.get('instruction', ''))]).strip()
        target_text = " ".join([str(raw_text.get('telugu_instruction', ''))]).strip()

        source_text_encoded = self.tokenizer_en.encode(source_text).ids
        target_text_encoded = self.tokenizer_te.encode(target_text).ids

        CLS_ID = torch.tensor([self.tokenizer_te.token_to_id("[CLS]")], dtype=torch.int64)
        SEP_ID = torch.tensor([self.tokenizer_te.token_to_id("[SEP]")], dtype=torch.int64)
        PAD_ID = torch.tensor([self.tokenizer_te.token_to_id("[PAD]")], dtype=torch.int64)

        num_source_padding = self.max_seq_len - len(source_text_encoded) - 2
        num_target_padding = self.max_seq_len - len(target_text_encoded) - 1

        encoder_padding = torch.tensor([PAD_ID] * num_source_padding, dtype=torch.int64)
        decoder_padding = torch.tensor([PAD_ID] * num_target_padding, dtype=torch.int64)

        encoder_input = torch.cat([
            CLS_ID,
            torch.tensor(source_text_encoded, dtype=torch.int64),
            SEP_ID,
            encoder_padding
        ], dim=0)

        decoder_input = torch.cat([
            CLS_ID,
            torch.tensor(target_text_encoded, dtype=torch.int64),
            decoder_padding
        ], dim=0)

        target_label = torch.cat([
            torch.tensor(target_text_encoded, dtype=torch.int64),
            SEP_ID,
            decoder_padding
        ], dim=0)

        encoder_input = encoder_input[:self.max_seq_len]
        decoder_input = decoder_input[:self.max_seq_len]
        target_label = target_label[:self.max_seq_len]

        encoder_mask = (encoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'target_label': target_label,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'source_text': source_text,
            'target_text': target_text
        }

def causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def create_dataloaders(train_dataset, val_dataset, batch_size: int):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader 