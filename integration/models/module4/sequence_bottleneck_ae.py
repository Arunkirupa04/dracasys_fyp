"""
SequenceBottleneckAE and SinusoidalPE — extracted from
module4/notebook/final/kaggle-source/mdc_model_vNext_kaggle.ipynb (cell 12).

DO NOT MODIFY. Must stay byte-for-byte identical to the training definition
so that model_hpo_best_vnext.pt loads without shape mismatch.
"""
import math
import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class SequenceBottleneckAE(nn.Module):
    def __init__(
        self,
        n_features,
        d_model=64,
        nhead=4,
        num_enc_layers=3,
        num_dec_layers=2,
        dim_ff=192,
        dropout=0.1,
        bottleneck_dim=16,
        max_len=200,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pe = SinusoidalPE(d_model, max_len=max_len, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)
        self.bn_down = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.bn_up = nn.Linear(bottleneck_dim, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.output_proj = nn.Linear(d_model, n_features)
        self.pos_queries = nn.Parameter(torch.zeros(max_len, d_model))
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

    def encode(self, x):
        return self.bn_down(self.encoder(self.pe(self.input_proj(x))))

    def forward(self, x):
        B, T, _ = x.shape
        z = self.encode(x)
        memory = self.bn_up(z)
        tgt = self.pe(self.pos_queries[:T].unsqueeze(0).expand(B, -1, -1))
        return self.output_proj(self.decoder(tgt, memory))
