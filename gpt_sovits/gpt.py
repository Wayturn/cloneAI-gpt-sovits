# 摘錄自 GPT-SoVITS v3 module/models.py
# 僅保留 GPT 類別與必要依賴
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, n_vocab=106, n_embd=256, n_layer=8, n_head=8):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=n_embd,
            nhead=n_head,
            num_encoder_layers=0,
            num_decoder_layers=n_layer,
            dim_feedforward=4*n_embd,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.token_emb = nn.Embedding(n_vocab, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, n_embd))
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_embd)

    def forward(self, x, memory):
        # x: (batch, seq, embd), memory: (batch, seq, embd)
        x = self.token_emb(x) + self.pos_emb[:, :x.size(1), :]
        y = self.transformer(
            src=memory,
            tgt=x
        )
        y = self.ln_f(y)
        return self.head(y)

    def generate(self, prompt_semantics, inference_ids):
        # 這裡簡化為直接 forward
        return self.forward(inference_ids, prompt_semantics)
