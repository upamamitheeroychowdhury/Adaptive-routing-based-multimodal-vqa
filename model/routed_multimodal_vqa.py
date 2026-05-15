import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=768,
        num_layers=2,
        num_heads=8,
        dropout=0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
        )

    def forward(self, x, mask=None):
        """
        x:    [B, T, D]
        mask: [B, T], 1 = valid, 0 = pad
        """

        x = self.input_proj(x)

        if mask is None:
            return self.encoder(x)

        empty_samples = mask.sum(dim=1) == 0

        if empty_samples.any():
            mask = mask.clone()
            x = x.clone()

            mask[empty_samples, 0] = 1
            x[empty_samples, 0, :] = 0.0

        key_padding_mask = mask == 0

        encoded = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask,
        )

        if empty_samples.any():
            encoded[empty_samples] = 0.0

        return encoded


class QAGuidedAttention(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()

        self.token_proj = nn.Linear(hidden_dim, hidden_dim)
        self.qa_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, tokens, qa_vec, mask=None):
        """
        tokens: [B, T, H]
        qa_vec: [B, H]
        mask:   [B, T], 1 = valid, 0 = pad
        """

        t = self.token_proj(tokens)
        q = self.qa_proj(qa_vec).unsqueeze(1)

        joint = torch.tanh(t + q)
        logits = self.score(joint).squeeze(-1)

        empty_samples = None

        if mask is not None:
            empty_samples = mask.sum(dim=1) == 0

            if empty_samples.any():
                mask = mask.clone()
                mask[empty_samples, 0] = 1

            logits = logits.masked_fill(mask == 0, -1e9)

        attn = F.softmax(logits, dim=-1)

        vec = torch.sum(tokens * attn.unsqueeze(-1), dim=1)

        if empty_samples is not None and empty_samples.any():
            vec = vec.clone()
            attn = attn.clone()

            vec[empty_samples] = 0.0
            attn[empty_samples] = 0.0

        return vec, attn


class TemporalVCPTEncoder(nn.Module):
    def __init__(
        self,
        num_objects,
        hidden_dim=768,
        num_layers=1,
        num_heads=8,
        dropout=0.2,
        padding_idx=0,
    ):
        super().__init__()

        self.obj_embed = nn.Embedding(
            num_embeddings=num_objects,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vcpt_obj_ids, vcpt_frame_mask):
        """
        vcpt_obj_ids:    [B, T, K]
        vcpt_frame_mask: [B, T], 1 = valid, 0 = pad
        """

        emb = self.obj_embed(vcpt_obj_ids)  # [B, T, K, H]

        obj_mask = (vcpt_obj_ids != 0).float().unsqueeze(-1)

        summed = (emb * obj_mask).sum(dim=2)
        denom = obj_mask.sum(dim=2).clamp(min=1.0)

        frame_tokens = summed / denom
        frame_tokens = self.norm(frame_tokens)

        empty_samples = vcpt_frame_mask.sum(dim=1) == 0

        if empty_samples.any():
            vcpt_frame_mask = vcpt_frame_mask.clone()
            frame_tokens = frame_tokens.clone()

            vcpt_frame_mask[empty_samples, 0] = 1
            frame_tokens[empty_samples, 0, :] = 0.0

        key_padding_mask = vcpt_frame_mask == 0

        encoded = self.encoder(
            frame_tokens,
            src_key_padding_mask=key_padding_mask,
        )

        if empty_samples.any():
            encoded[empty_samples] = 0.0

        return encoded


class ModalityRouter(nn.Module):
    def __init__(self, hidden_dim=768, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, qa_vec, video_vec, sub_vec, vcpt_vec):
        """
        Router output order:
        0 = video
        1 = subtitle
        2 = VCPT
        """

        x = torch.cat(
            [qa_vec, video_vec, sub_vec, vcpt_vec],
            dim=-1,
        )

        logits = self.net(x)

        weights = F.softmax(logits, dim=-1)

        return weights


class RoutedMultimodalVQA(nn.Module):
    def __init__(
        self,
        num_objects,
        bert_name="bert-base-uncased",
        siglip_dim=768,
        hidden_dim=768,
        dropout=0.2,
        freeze_bert=False,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_name)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.video_encoder = TemporalTransformer(
            input_dim=siglip_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=8,
            dropout=dropout,
        )

        self.vcpt_encoder = TemporalVCPTEncoder(
            num_objects=num_objects,
            hidden_dim=hidden_dim,
            num_layers=1,
            num_heads=8,
            dropout=dropout,
        )

        self.video_qa_attention = QAGuidedAttention(hidden_dim)
        self.vcpt_qa_attention = QAGuidedAttention(hidden_dim)

        self.router = ModalityRouter(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.answer_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_text(self, input_ids, attention_mask):
        """
        input_ids:
            [B, L] or [B, 5, L]
        """

        if input_ids.dim() == 3:
            B, A, L = input_ids.shape

            input_ids = input_ids.reshape(B * A, L)
            attention_mask = attention_mask.reshape(B * A, L)

            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            cls = out.last_hidden_state[:, 0, :]
            cls = cls.reshape(B, A, -1)

            return cls

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return out.last_hidden_state[:, 0, :]

    def forward(
        self,
        qa_input_ids,
        qa_attention_mask,
        sub_input_ids,
        sub_attention_mask,
        video_feats,
        video_mask,
        vcpt_obj_ids,
        vcpt_obj_mask,
    ):
        """
        qa_input_ids:   [B, 5, L]
        sub_input_ids:  [B, L]
        video_feats:    [B, T, D]
        video_mask:     [B, T]
        vcpt_obj_ids:   [B, T, K]
        vcpt_obj_mask:  [B, T]
        """

        qa_vecs = self.encode_text(
            qa_input_ids,
            qa_attention_mask,
        )

        sub_vec = self.encode_text(
            sub_input_ids,
            sub_attention_mask,
        )

        video_tokens = self.video_encoder(
            video_feats,
            video_mask,
        )

        vcpt_tokens = self.vcpt_encoder(
            vcpt_obj_ids,
            vcpt_obj_mask,
        )

        B, A, H = qa_vecs.shape

        all_scores = []
        all_router_weights = []
        all_video_attention = []
        all_vcpt_attention = []

        for i in range(A):
            qa_i = qa_vecs[:, i, :]

            video_vec, video_attn = self.video_qa_attention(
                video_tokens,
                qa_i,
                video_mask,
            )

            vcpt_vec, vcpt_attn = self.vcpt_qa_attention(
                vcpt_tokens,
                qa_i,
                vcpt_obj_mask,
            )

            router_weights = self.router(
                qa_i,
                video_vec,
                sub_vec,
                vcpt_vec,
            )

            w_video = router_weights[:, 0].unsqueeze(-1)
            w_sub = router_weights[:, 1].unsqueeze(-1)
            w_vcpt = router_weights[:, 2].unsqueeze(-1)

            fused = (
                w_video * video_vec
                + w_sub * sub_vec
                + w_vcpt * vcpt_vec
            )

            fused = self.fusion_norm(fused)

            answer_input = torch.cat(
                [qa_i, fused],
                dim=-1,
            )

            score = self.answer_head(answer_input).squeeze(-1)

            all_scores.append(score)
            all_router_weights.append(router_weights)
            all_video_attention.append(video_attn)
            all_vcpt_attention.append(vcpt_attn)

        scores = torch.stack(all_scores, dim=1)
        router_weights = torch.stack(all_router_weights, dim=1)
        video_attention = torch.stack(all_video_attention, dim=1)
        vcpt_attention = torch.stack(all_vcpt_attention, dim=1)

        return {
            "scores": scores,
            "router_weights": router_weights,
            "video_attention": video_attention,
            "vcpt_attention": vcpt_attention,
        }