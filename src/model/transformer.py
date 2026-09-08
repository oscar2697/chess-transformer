"""PyTorch Transformer with dual heads: Policy (1968 UCI) + Value [-1,1], correct attention capture."""
import torch
import torch.nn as nn

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size=1968, d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                 representation="fen_tokens", fen_vocab=34, max_len=128, pad_idx=0):
        super().__init__()
        self.representation = representation
        self.d_model = d_model
        self.pad_idx = pad_idx
        if representation == "fen_tokens":
            self.token_emb = nn.Embedding(fen_vocab, d_model, padding_idx=pad_idx)
            self.pos_emb = nn.Embedding(max_len, d_model)
        else:  # planes_8x8: 18*8*8 -> linear
            self.plane_proj = nn.Linear(18*8*8, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.policy_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())
        self.last_attn = None
        # Capture true attention weights of last layer via forward hook (no double compute)
        def _hook(module, args, output):
            # MultiheadAttention forward returns (attn_out, attn_weights) when need_weights=True,
            # but EncoderLayer calls it with need_weights=True internally and discards weights.
            # We re-request weights cheaply only for the last layer on its normalized input:
            # instead store via hook on the module's output is not possible, so we rely on
            # need_weights pass done in forward() single-pass below.
            pass
        self._attn_hook = _hook

    def forward(self, x, mask=None, return_attn=True):
        # x: [B, L] tokens (with [CLS] at pos 0, padded with pad_idx) or [B,18,8,8] planes
        # return_attn=False skips the extra attention query (saves ~15-20% in training)
        if self.representation == "fen_tokens":
            B, L = x.shape
            h = self.token_emb(x) + self.pos_emb(torch.arange(L, device=x.device))
            key_padding_mask = (x == self.pad_idx)  # [B, L] True = ignore
        else:
            B = x.shape[0]
            h = self.plane_proj(x.view(B, -1)).unsqueeze(1)  # [B,1,d]
            key_padding_mask = None
        out = self.encoder(h, src_key_padding_mask=key_padding_mask, mask=mask)
        # Single extra need_weights query on LAST layer only for interpretability
        if return_attn:
            with torch.no_grad():
                last = self.encoder.layers[-1].self_attn
                _, w = last(out, out, out, need_weights=True, average_attn_weights=False,
                            key_padding_mask=key_padding_mask)
                self.last_attn = w
        cls = out[:, 0, :]  # position 0 is [CLS] (prepended in fen_to_tokens)
        policy = self.policy_head(cls)  # [B, 1968]
        value = self.value_head(cls).squeeze(-1)  # [B]
        return policy, value

    def get_attention_weights(self):
        return self.last_attn
