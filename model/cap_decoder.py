"""
Captioning Decoder.

Multi-granularity memory:
  - global_token: GAP(F_cap @ 1/32)               -> 1 token   (overall context)
  - meso_tokens : pooled(F_cap @ 1/8)             -> K tokens  (region-level)
  - local_tokens: flatten(F_cap @ 1/16)           -> HW tokens (spatial detail)
Memory = [global; meso; local]

A lightweight Transformer decoder performs masked self-attention over the
caption prefix and cross-attention over the memory to produce next-token logits.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CapDecoder(nn.Module):
    def __init__(self, d_model=256, vocab_size=1000, max_length=41,
                 word_vocab=None, n_head=8, n_layers=1, dropout=0.1,
                 use_meso=True, meso_pool_size=2, num_classes=3):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_vocab = word_vocab
        self.use_meso = use_meso

        # Global token projection
        self.global_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True))

        # Local token projection (F_cap @ 1/16)
        self.local_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False), nn.BatchNorm2d(d_model), nn.ReLU(inplace=True))

        # Meso token projection (F_cap @ 1/8)
        if use_meso:
            self.meso_pool_size = meso_pool_size
            self.meso_proj = nn.Sequential(
                nn.Conv2d(d_model, d_model, 1, bias=False), nn.BatchNorm2d(d_model), nn.ReLU(inplace=True))
            print(f'[CapDecoder] Meso tokens enabled (pool={meso_pool_size}x{meso_pool_size})')

        # Word embedding + positional encoding
        self.vocab_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_length)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        self._init_weights()

    def _init_weights(self):
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)
        self.output_proj.bias.data.fill_(0)
        self.output_proj.weight.data.uniform_(-0.1, 0.1)

    def _build_memory(self, chg_feats):
        chg_8  = chg_feats[1]   # (B, C, H_8, W_8)
        chg_16 = chg_feats[2]   # (B, C, H_16, W_16)
        chg_32 = chg_feats[3]   # (B, C, H_32, W_32)
        B, C, H16, W16 = chg_16.shape

        # Global token (mean instead of adaptive_avg_pool2d for deterministic backward)
        global_token = chg_32.mean(dim=[2, 3])                   # (B, C)
        global_token = self.global_proj(global_token).unsqueeze(0)  # (1, B, C)

        # Meso tokens
        meso_tokens = None
        if self.use_meso:
            # avg_pool2d with computed kernel (deterministic backward, unlike adaptive_avg_pool2d)
            kH = chg_8.shape[2] // self.meso_pool_size
            kW = chg_8.shape[3] // self.meso_pool_size
            meso = F.avg_pool2d(chg_8, kernel_size=(kH, kW), stride=(kH, kW))
            meso = self.meso_proj(meso)
            meso_tokens = meso.view(B, C, -1).permute(2, 0, 1)   # (K, B, C)

        # Local tokens
        local_feat = self.local_proj(chg_16)
        local_tokens = local_feat.view(B, C, -1).permute(2, 0, 1)  # (HW, B, C)

        # Compose memory: [global; meso; local]
        parts = [global_token]
        if meso_tokens is not None:
            parts.append(meso_tokens)
        parts.append(local_tokens)
        return torch.cat(parts, dim=0)

    def forward(self, chg_feats, encoded_captions, caption_lengths):
        memory = self._build_memory(chg_feats)

        word_length = encoded_captions.size(1)
        mask = torch.triu(torch.ones(word_length, word_length,
                          device=encoded_captions.device) * float('-inf'), diagonal=1)
        pad_mask = (encoded_captions == self.word_vocab['<NULL>']) | \
                   (encoded_captions == self.word_vocab['<END>'])

        word_emb = self.vocab_embedding(encoded_captions).transpose(1, 0)
        word_emb = self.position_encoding(word_emb)

        cap_hidden = self.transformer(word_emb, memory, tgt_mask=mask,
                                      tgt_key_padding_mask=pad_mask)
        pred = self.output_proj(self.dropout_layer(cap_hidden)).permute(1, 0, 2)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        cap_hidden = cap_hidden[:, sort_ind, :]
        decode_lengths = (caption_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind, cap_hidden

    @torch.no_grad()
    def sample(self, chg_feats):
        memory = self._build_memory(chg_feats)
        device = memory.device

        tgt = torch.zeros(1, self.max_length, dtype=torch.long, device=device)
        mask = torch.triu(torch.ones(self.max_length, self.max_length, device=device)
                          * float('-inf'), diagonal=1)
        tgt[0, 0] = self.word_vocab['<START>']
        seqs = [self.word_vocab['<START>']]

        for step in range(self.max_length):
            pad_mask = (tgt == self.word_vocab['<NULL>']) | (tgt == self.word_vocab['<END>'])
            word_emb = self.vocab_embedding(tgt).transpose(1, 0)
            word_emb = self.position_encoding(word_emb)
            out = self.transformer(word_emb, memory, tgt_mask=mask, tgt_key_padding_mask=pad_mask)
            logits = self.output_proj(out)
            predicted_id = logits[step, 0].argmax().item()
            seqs.append(predicted_id)
            if predicted_id == self.word_vocab['<END>']: break
            if step < self.max_length - 1: tgt[0, step + 1] = predicted_id

        return seqs
