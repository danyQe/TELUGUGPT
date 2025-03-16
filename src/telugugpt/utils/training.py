import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from typing import Callable, Dict, Any
from ..config.config import DEVICE, TRAINING_CONFIG

def train_model(model: nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                val_dataloader: torch.utils.data.DataLoader,
                tokenizer_en,
                tokenizer_te,
                checkpoint_path: str,
                preload_epoch: int = None) -> None:
    """Train the transformer model."""
    
    initial_epoch = 0
    global_step = 0
    
    optimizer = Adam(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        eps=1e-9
    )

    if preload_epoch is not None:
        model_filename = f"{checkpoint_path}/model_{preload_epoch}.pt"
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_en.token_to_id('[PAD]'),
        label_smoothing=TRAINING_CONFIG["label_smoothing"]
    ).to(DEVICE)

    for epoch in range(initial_epoch, TRAINING_CONFIG["epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch_idx, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(DEVICE)
            decoder_input = batch['decoder_input'].to(DEVICE)
            encoder_mask = batch['encoder_mask'].to(DEVICE)
            decoder_mask = batch['decoder_mask'].to(DEVICE)
            target_label = batch['target_label'].to(DEVICE)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            projection_output = model.project(decoder_output)

            loss = loss_fn(
                projection_output.view(-1, tokenizer_te.get_vocab_size()),
                target_label.view(-1)
            )
            
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            loss = loss / TRAINING_CONFIG["gradient_accumulation_steps"]
            loss.backward()
            
            if (batch_idx + 1) % TRAINING_CONFIG["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Save checkpoint
        model_filename = f"{checkpoint_path}/model_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        # Run validation
        run_validation(model, val_dataloader, tokenizer_en, tokenizer_te, DEVICE)

def run_validation(model: nn.Module,
                  validation_ds: torch.utils.data.DataLoader,
                  tokenizer_en,
                  tokenizer_te,
                  device: torch.device,
                  num_examples: int = 2) -> None:
    """Run validation and print example translations."""
    
    model.eval()
    count = 0

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            cls_id = tokenizer_te.token_to_id('[CLS]')
            sep_id = tokenizer_te.token_to_id('[SEP]')

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_input = torch.empty(1, 1).fill_(cls_id).type_as(encoder_input).to(device)

            while True:
                if decoder_input.size(1) == model.max_seq_len:
                    break

                decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
                out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                prob = model.project(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                
                decoder_input = torch.cat([
                    decoder_input,
                    torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)
                ], dim=1)

                if next_word == sep_id:
                    break

            model_out = decoder_input.squeeze(0)
            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = tokenizer_te.decode(model_out.detach().cpu().numpy())

            print('-' * 55)
            print(f'Source Text: {source_text}')
            print(f'Target Text: {target_text}')
            print(f'Predicted: {model_out_text}')

            if count == num_examples:
                break

def causal_mask(size: int) -> torch.Tensor:
    """Create causal mask for decoder self-attention."""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0 