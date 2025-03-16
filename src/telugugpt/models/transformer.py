import torch
import torch.nn as nn
from .layers import (
    MultiHeadAttention,
    FeedForward,
    LayerNorm,
    AddAndNorm,
    EmbeddingLayer,
    PositionalEncoding
)

class EncoderBlock(nn.Module):
    def __init__(self, multihead_attention: MultiHeadAttention, feed_forward: FeedForward, dropout_rate: float):
        super().__init__()
        self.multihead_attention = multihead_attention
        self.feed_forward = feed_forward
        self.addnorm_1 = AddAndNorm(dropout_rate)
        self.addnorm_2 = AddAndNorm(dropout_rate)

    def forward(self, encoder_input, encoder_mask):
        encoder_input = self.addnorm_1(
            encoder_input,
            lambda x: self.multihead_attention(x, x, x, encoder_mask)
        )
        encoder_input = self.addnorm_2(encoder_input, self.feed_forward)
        return encoder_input

class Encoder(nn.Module):
    def __init__(self, encoderblocklist: nn.ModuleList):
        super().__init__()
        self.encoderblocklist = encoderblocklist
        self.layer_norm = LayerNorm()

    def forward(self, encoder_input, encoder_mask):
        for encoderblock in self.encoderblocklist:
            encoder_input = encoderblock(encoder_input, encoder_mask)
        return self.layer_norm(encoder_input)

class DecoderBlock(nn.Module):
    def __init__(self, masked_multihead_attention: MultiHeadAttention,
                 cross_multihead_attention: MultiHeadAttention,
                 feed_forward: FeedForward, dropout_rate: float):
        super().__init__()
        self.masked_multihead_attention = masked_multihead_attention
        self.cross_multihead_attention = cross_multihead_attention
        self.feed_forward = feed_forward
        self.addnorm_1 = AddAndNorm(dropout_rate)
        self.addnorm_2 = AddAndNorm(dropout_rate)
        self.addnorm_3 = AddAndNorm(dropout_rate)

    def forward(self, decoder_input, encoder_output, encoder_mask, decoder_mask):
        decoder_input = self.addnorm_1(
            decoder_input,
            lambda x: self.masked_multihead_attention(x, x, x, decoder_mask)
        )
        decoder_input = self.addnorm_2(
            decoder_input,
            lambda x: self.cross_multihead_attention(x, encoder_output, encoder_output, encoder_mask)
        )
        decoder_input = self.addnorm_3(decoder_input, self.feed_forward)
        return decoder_input

class Decoder(nn.Module):
    def __init__(self, decoderblocklist: nn.ModuleList):
        super().__init__()
        self.decoderblocklist = decoderblocklist
        self.layer_norm = LayerNorm()

    def forward(self, decoder_input, encoder_output, encoder_mask, decoder_mask):
        for decoderblock in self.decoderblocklist:
            decoder_input = decoderblock(decoder_input, encoder_output, encoder_mask, decoder_mask)
        return self.layer_norm(decoder_input)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection_layer = nn.Linear(d_model, vocab_size)

    def forward(self, decoder_output):
        return self.projection_layer(decoder_output)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 source_embed: EmbeddingLayer, target_embed: EmbeddingLayer,
                 source_pos: PositionalEncoding, target_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, encoder_input, encoder_mask):
        encoder_input = self.source_embed(encoder_input)
        encoder_input = self.source_pos(encoder_input)
        return self.encoder(encoder_input, encoder_mask)

    def decode(self, encoder_output, encoder_mask, decoder_input, decoder_mask):
        decoder_input = self.target_embed(decoder_input)
        decoder_input = self.target_pos(decoder_input)
        return self.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)

    def project(self, decoder_output):
        return self.projection_layer(decoder_output)

def build_transformer(source_vocab_size: int, target_vocab_size: int,
                     source_seq_len: int, target_seq_len: int,
                     d_model: int = 512, num_blocks: int = 6,
                     num_heads: int = 8, dropout_rate: float = 0.1,
                     d_ff: int = 2048) -> Transformer:
    
    source_embed = EmbeddingLayer(d_model, source_vocab_size)
    target_embed = EmbeddingLayer(d_model, target_vocab_size)

    source_pos = PositionalEncoding(d_model, source_seq_len, dropout_rate)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout_rate)

    encoderblocklist = []
    for _ in range(num_blocks):
        multihead_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        encoder_block = EncoderBlock(multihead_attention, feed_forward, dropout_rate)
        encoderblocklist.append(encoder_block)

    decoderblocklist = []
    for _ in range(num_blocks):
        masked_multihead_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        cross_multihead_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        decoder_block = DecoderBlock(masked_multihead_attention, cross_multihead_attention,
                                   feed_forward, dropout_rate)
        decoderblocklist.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoderblocklist))
    decoder = Decoder(nn.ModuleList(decoderblocklist))
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    model = Transformer(encoder, decoder, source_embed, target_embed,
                       source_pos, target_pos, projection_layer)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model 