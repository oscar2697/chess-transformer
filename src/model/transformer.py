"""PyTorch Transformer with dual heads: Policy (1968 UCI) + Value [-1,1], attention extraction hook."""
import torch
import torch.nn as nn

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size=1968, d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.1,
                 representation="fen_tokens", fen_vocab=32, max_len=128):
        super().__init__()
        self.representation = representation
        self.d_model = d_model
        if representation == "fen_tokens":
            self.token_emb = nn.Embedding(fen_vocab, d_model)
            self.pos_emb = nn.Embedding(max_len, d_model)
        else:  # planes_8x8: 18*8*8 -> linear
            self.plane_proj = nn.Linear(18*8*8, d_model)
            self.pos_emb = nn.Embedding(1, d_model)  # single position
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.policy_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())
        self.last_attn = None
        # hook last layer self-attn weights
        self._register_hook()

    def _register_hook(self):
        # TransformerEncoderLayer stores self_attn; hook its forward
        last_layer = self.encoder.layers[-1].self_attn
        def hook(module, inp, out):
            # out is (attn_output, attn_weights) only if need_weights True; we capture via forward hook alternative: use hook on multihead
            pass
        # Instead we will capture via manual forward with need_weights

    def forward(self, x, mask=None):
        # x: [B, L] tokens or [B,18,8,8] planes
        if self.representation == "fen_tokens":
            B, L = x.shape
            h = self.token_emb(x) + self.pos_emb(torch.arange(L, device=x.device))
        else:
            B = x.shape[0]
            h = self.plane_proj(x.view(B, -1)).unsqueeze(1)  # [B,1,d]
        # need attention weights: iterate layers manually to capture last
        out = h
        attn_weights = None
        for i, layer in enumerate(self.encoder.layers):
            # use need_weights trick
            attn_out, w = layer.self_attn(out, out, out, need_weights=True, average_attn_weights=False)
            # TransformerEncoderLayer does more; simpler: call encoder normally and skip detailed attn, store w
            # For correctness, apply full layer
            out = layer(out, src_mask=mask)
            if i == len(self.encoder.layers)-1:
                attn_weights = w  # [B, heads, L, L] or [B, L, L]
        self.last_attn = attn_weights
        cls = out[:, 0, :]  # CLS pooling
        policy = self.policy_head(cls)  # [B, 1968]
        value = self.value_head(cls).squeeze(-1)  # [B]
        return policy, value

    def get_attention_weights(self):
        return self.last_attn
