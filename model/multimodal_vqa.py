import torch
import torch.nn as nn
from transformers import BertModel


class MultimodalVQA(nn.Module):
    def __init__(self, bert_name="bert-base-uncased", hidden_dim=768, dropout=0.2):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_name)

        self.frame_proj = nn.Linear(768, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, frame_feats):
        batch_size, num_choices, seq_len = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * num_choices, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_choices, seq_len)

        text_outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask
        )

        text_cls = text_outputs.last_hidden_state[:, 0, :]

        frame_feats = frame_feats.unsqueeze(1).repeat(1, num_choices, 1, 1)

        frame_feats = frame_feats.view(
            batch_size * num_choices,
            frame_feats.size(2),
            frame_feats.size(3)
        )

        frame_feats = self.frame_proj(frame_feats)

        query = text_cls.unsqueeze(1)

        attended_video, _ = self.cross_attn(
            query=query,
            key=frame_feats,
            value=frame_feats
        )

        fused = self.norm1(query + attended_video)
        fused = self.norm2(fused + self.ffn(fused))

        fused = fused.squeeze(1)
        fused = self.dropout(fused)

        scores = self.classifier(fused)
        scores = scores.view(batch_size, num_choices)

        return scores