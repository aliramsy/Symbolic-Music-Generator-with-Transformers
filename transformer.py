import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def sinusoidal_position_encoding(num_positions, d_model):
    angles = _get_angles(
        np.arange(num_positions)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )
    sines = np.sin(angles[:, 0::2])
    cosines = np.cos(angles[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)


def _get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        input_vocab_size,
        target_vocab_size,
        max_num_positions_in_pe_encoder,
        max_num_positions_in_pe_decoder,
        dropout_rate=0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            d_feedforward,
            input_vocab_size,
            max_num_positions_in_pe_encoder,
            dropout_rate,
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            d_feedforward,
            target_vocab_size,
            max_num_positions_in_pe_decoder,
            dropout_rate,
        )
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, input, target, enc_padding_mask, look_ahead_mask, dec_padding_mask, attn_mask):
        enc_output = self.encoder(input, enc_padding_mask)
        dec_output = self.decoder(
            target, enc_output, look_ahead_mask, dec_padding_mask, attn_mask)
        logits = self.final_layer(dec_output)
        return logits


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, input_vocab_size, maximum_positions_in_pe, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = sinusoidal_position_encoding(
            maximum_positions_in_pe, d_model)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_feedforward, dropout_rate)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        x = self.embedding(
            x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, target_vocab_size, maximum_positions_in_pe, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = sinusoidal_position_encoding(
            maximum_positions_in_pe, d_model)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_feedforward, dropout_rate)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask, attn_mask):
        x = self.embedding(
            x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask, attn_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, key_padding_mask=mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output))
        return out2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask, attn_mask):
        attn1, _ = self.mha1(x, x, x, attn_mask=attn_mask)
        # attn1, _ = self.mha1(x, x, x, key_padding_mask=look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout1(attn1))
        attn2, _ = self.mha2(out1, enc_output, enc_output,
                             key_padding_mask=padding_mask)
        out2 = self.layernorm2(out1 + self.dropout2(attn2))
        ffn_output = self.ffn(out2)
        out3 = self.layernorm3(out2 + self.dropout3(ffn_output))
        return out3


if __name__ == "__main__":
    num_layers = 2
    d_model = 64
    num_heads = 2
    d_feedforward = 128
    input_vocab_size = 100
    target_vocab_size = 100
    dropout_rate = 0.1
    pe_input = 10
    pe_target = 10

    transformer_model = Transformer(
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        dropout_rate,
    )

    dummy_inp = torch.randint(0, input_vocab_size, (1, 10))
    dummy_tar = torch.randint(0, target_vocab_size, (1, 10))

    logits = transformer_model(
        dummy_inp, dummy_tar, None, None, None
    )
    print(logits.shape)
